# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/


"""Distill pretraind diffusion-based generative model using the techniques described in the
paper "Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation"."""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import google3_util_metric
import socket
import portpicker
import re
import json
import click
import torch
import dnnlib


import time
import copy
import pickle
import psutil
import PIL.Image
import numpy as np




from torch_utils import distributed as dist

from torch_utils import training_stats
from torch_utils import misc

from metrics import sid_metric_main as metric_main

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

    
class CommaSeparatedList(click.ParamType):
    
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')
    
@google3_util_metric.google3_metric
def calculate_metric(metric, metric_pt_path, G, init_sigma, dataset_kwargs, num_gpus, rank, device):
    return metric_main.calc_metric(metric=metric, metric_pt_path=metric_pt_path, G=G, init_sigma=init_sigma,
        dataset_kwargs=dataset_kwargs, num_gpus=num_gpus, rank=rank, device=device)

@google3_util_metric.google3_append
def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')
        
@google3_util_metric.google3_save
def save_metric(result_dict, fname):        
    # formatted_string = f'{metric:g}'  # Format the number using :g specifier
    # # Open the file in write mode and save the formatted string
    # with open(fname, 'w') as file:
    #     file.write(formatted_string)  
    with open(fname, "w") as file:
        for key, value in result_dict.items():
            file.write(f"{key}: {value}\n")



#----------------------------------------------------------------------------


@click.command()
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--init_sigma',    help='Noise standard deviation that is fixed during distillation and generation', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=2.5, show_default=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)

# Main options.gpu
# Adapted from Diff-Instruct
@click.option('--metrics',       help='Comma-separated list or "none" [default: fid50k_full]',      type=CommaSeparatedList())
# FID metric PT path
@click.option('--metric_pt_path',  help='Where the metric pt locates on Google filesystem: CNS',     metavar='DIR', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt', show_default=True)
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)

@google3_util_metric.google3_torchrun
def torchrun(**kwargs):
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(
      'Found %d accelerator(s). Spawning data parallel workers.' % world_size
    )
    kwargs['master_port'] = portpicker.pick_unused_port()
    torch.multiprocessing.start_processes(
      main, args=[kwargs], nprocs=world_size, join=True, start_method='spawn'
  )


def main(local_rank, kwargs):
    """Distill pretraind diffusion-based generative model using the techniques described in the
paper "Score identity Distillation: Exponentially Fast Distillation of
Pretrained Diffusion Models for One-Step Generation".
    
    Examples:

    \b
    # Distill EDM model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 sid_train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    os.environ['MASTER_ADDR'] = socket.gethostname()
    os.environ['MASTER_PORT'] = str(kwargs['master_port'])
    os.environ['RANK'] = str(local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
    dist.init()

    google3_util_metric.google3_cns_init(opts, dist)

    network_pkl=opts.network_pkl
    metric_pt_path = opts.metric_pt_path
    metrics = opts.metrics
    init_sigma = opts.init_sigma
    #google3_util_metric.google3_cns_init(opts, dist)
    
    
    #if network_pkl is None:
    #    network_pkl=opts.resume
    
    
    #seeds = opts.seeds
    #max_batch_size = opts.max_batch_size
    # image_outdir = opts.image_outdir
    
    #num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    #all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    #rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    device = torch.device('cuda')

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        G_ema = pickle.load(f)['ema'].to(device)
    
    dist.print0(f'Finished loading "{network_pkl}"...')
    
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond)
    dataset_kwargs.resolution = G_ema.img_resolution
    #network_kwargs.use_fp16=0
    #network_kwargs.use_labels=0
    dataset_kwargs.max_size=2000000
    dist.print0(dataset_kwargs)
        
    #, xflip=opts.xflip, cache=opts.cache)
    #dataset_kwargs={}
    
    match = re.search(r"-(\d+)\.pkl$", network_pkl)
    if match:
        # If a match is found, extract the number part
        number_part = match.group(1)
    else:
        # If no match is found, handle the situation (e.g., use a default value or skip processing)
        number_part = '_final'  # Or any other handling logic you prefer
        
    for metric in metrics:
        dist.print0(metric)
        result_dict = calculate_metric(metric=metric, metric_pt_path=metric_pt_path, G=G_ema, init_sigma=init_sigma,
            dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
        if dist.get_rank() == 0:
            print(result_dict.results)
            txt_path = os.path.join(os.path.dirname(opts.outdir),f'{metric}{number_part}.txt')
            print(txt_path)
            save_metric(result_dict=result_dict,fname=txt_path)
            #metric_main.report_metric(result_dict, run_dir=opts.outdir) 
            #metric_main.report_metric(result_dict, run_dir=opts.outdir, snapshot_pkl='fakes_metric.png', alpha=alpha)                        
            #metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{alpha:03f}_{0//1000:06d}.png', alpha=alpha)
            #metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes_{alpha:03f}_{cur_nimg//1000:06d}.png', alpha=alpha)          
        #stats_metrics.update(result_dict.results)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    google3_util_metric.google3_run(torchrun)

#----------------------------------------------------------------------------
