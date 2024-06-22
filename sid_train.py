# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


"""Distill Stable Diffusion models using the SiD-LSG techniques described in the
paper "Long and Short Guidance in Score identity Distillation for One-Step Text-to-Image Generation"."""

import os
import socket
import re
import json
import click
import torch
import dnnlib
import timm
import wcwidth
import ftfy

from torch_utils import distributed as dist
from training import sid_training_loop as training_loop
import sys

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


def find_latest_checkpoint(directory):
    """
    Finds the latest training state checkpoint file in a directory and its subdirectories.

    :param directory: The path to the directory to search in.
    :param os: The package to use to perform file operations.
    :return: The path to the latest checkpoint file, or None if no such file is found.
    """
    latest_file = None
    latest_number = -1
    print(directory)
    for root, dirs, files in os.walk(directory):  
        print(root)
        print(files)
        for file in files:
            if file.startswith("training-state-") and file.endswith(".pt"):
                # Extract the number from the file name
                number_part = file[len("training-state-"):-len(".pt")]
                try:
                    number = int(number_part)
                    if number > latest_number:
                        latest_number = number
                        latest_file = os.path.join(root, file)
                except ValueError:
                    # If the number part is not an integer, ignore this file
                    continue
    print(latest_file)
    return latest_file, latest_number

    
class CommaSeparatedList(click.ParamType):
    
    name = 'list'
    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------


@click.command()

# Main options.gpu
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=False)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--data_stat',     help='Path to the dataset stats', metavar='ZIP|DIR',               type=str, default=None)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='FLOAT',                    type=float, default=0.0, show_default=True)


# Performance-related.
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)


# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)
@click.option('--metrics',       help='Comma-separated list or "none" [fid30k_full, fid_clip_30k_full, fid_test, or fid_clip_test]',  type=CommaSeparatedList(),  default=None)
@click.option('--sd_model',      help='sd_model',                                                    type=str, default="runwayml/stable-diffusion-v1-5")
@click.option('--resolution',    help='Image resolution', metavar='INT',                             type=int, default=512, show_default=True)


# Parameters for SiD
@click.option('--init_timestep', help='t_init, in [0,999]', metavar='INT', type=int, default=625, show_default=True)
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--lsg',           help='Loss scaling G', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--alpha',         help='L2-alpha*L1', metavar='FLOAT',                               type=click.FloatRange(min=-1000, min_open=True), default=1, show_default=True)
@click.option('--tmax',          help='The largest allowed time step when evaluating the teacher model, in [0,1000]', metavar='INT',  type=click.IntRange(min=0), default=980, show_default=True)
@click.option('--tmin',          help='The smallest allowed time step when evaluating the teacher model, in [0,1000]', metavar='INT',  type=click.IntRange(min=0), default=20, show_default=True)
@click.option('--lr',            help='Learning rate of fake score estimation network', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1e-6, show_default=True)
@click.option('--glr',           help='Learning rate of fake data generator', metavar='FLOAT',      type=click.FloatRange(min=0, min_open=True), default=1e-6, show_default=True)

@click.option('--train_mode',    help='Distill generator or eval its metrics', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--network_pkl',   help='network pickle for calculating metrics', metavar='PKL|URL',  type=str,default=None)

# unqiue to SiD-LSG
@click.option('--cfg_train_fake',      help='kappa1, guidance scale in training fake. Default value is 1.0.', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--cfg_eval_fake',       help='kappa2=kappa3, guidance scale in evaluating fake. Default value is 1.0. kappa2 and kappa3 could be different but we set them the same by default', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--cfg_eval_real',       help='kappa4, guidance scale in evaluating real. Default value is 1.0.', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--data_prompt_text',    help='Path to training prompts', metavar='ZIP|DIR',                     type=str, required=True)

# Model pathes for computing FID and CLIP scores
@click.option('--metric_pt_path',  help='Where the metric pt locates',     metavar='DIR', type=str, default='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt', show_default=True)
@click.option('--metric_clip_path',  help='Where the metric clip file locates',     metavar='DIR', type=str,default='clip-vit-base-patch16')  #not used by default; can be enabled by providing a user-defined CLIP encoder
@click.option('--metric_open_clip_path',  help='Where the metric open clip file locates',     metavar='DIR', type=str,default='clipvitg14.pkl')
 
@click.option('--enable_xformers',          help='Use xformers is it is available', metavar='BOOL',            type=bool, default=True, show_default=True)
@click.option('--gradient_checkpointing',   help='Use gradient_checkpointing to save memory, if necessary', metavar='BOOL',            type=bool, default=False, show_default=True)



#Options to be developed; default values will be used
@click.option('--optimizer',  help='Optimizer',     metavar='adam|adamw', type=str, default='adam', show_default=True)
@click.option('--num_steps', help='Number of generation steps (NFEs)', metavar='INT', type=int, default=1, show_default=True)
@click.option('--fake_score_use_lora',          help='Use lora for fake score estimation', metavar='BOOL',            type=bool, default=False, show_default=True)



def main(**kwargs):
    """Distill Stable Diffusion models using the SiD-LSG techniques described in the
paper "Long and Short Guidance in Score identity Distillation for One-Step Text-to-Image Generation".
    
    Examples:

    \b
    # Distill Stable Diffusion 1.5 model using 4 GPUs
    torchrun --standalone --nproc_per_node=4 sid_train.py \
    --outdir 'image_experiment/sid-lsg-train-runs/' \
    --data '/data/datasets/MS-COCO-256/val' \
    --train_mode 1 \
    --cfg_train_fake 2 \
    --cfg_eval_fake 2 \
    --cfg_eval_real 2 \
    --optimizer 'adam' \
    --data_prompt_text '/data/datasets/aesthetics_6_plus' \
    --resolution 512 \
    --alpha 1 \
    --init_timestep 625 \
    --batch 512 \
    --fp16 0 \
    --batch-gpu 1 \
    --sd_model "runwayml/stable-diffusion-v1-5" \
    --tick 2 \
    --snap 50 \
    --dump 100 \
    --lr 0.000001 \
    --glr 0.000001 \
    --duration 500 \
    --metrics 'fid_clip_30k_full' \
    --ema 0.05
    
    """
    
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    
    
    

    # Initialize config dict.
    c = dnnlib.EasyDict()
    
    c.metrics = opts.metrics
    
    c.resolution=opts.resolution
    
    
    
        
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    if opts.train_mode:
        c.dataset_prompt_text_kwargs = dnnlib.EasyDict(class_name='training.aesthetics_dataset.ImageDataset', path=opts.data_prompt_text, resolution=opts.resolution, random_flip=opts.xflip, prompt_only=True)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    
    if opts.optimizer=='adam':
        c.fake_score_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.0, 0.999], eps = 1e-8 if not opts.fp16 else 1e-6)
        c.g_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.glr, betas=[0.0, 0.999], eps = 1e-8 if not opts.fp16 else 1e-6)
    else:
        #this is another optimizer to choose; it could provide better performance, but we have not carefully tested it yet
        assert opts.optimizer=='adamw'
        c.fake_score_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.lr, betas=[0.0, 0.999], eps = 1e-8 if not opts.fp16 else 1e-6,weight_decay=0.01)
        c.g_optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.AdamW', lr=opts.glr, betas=[0.0, 0.999], eps = 1e-8 if not opts.fp16 else 1e-6,weight_decay=0.01)
    
    c.init_timestep = opts.init_timestep

    
    if opts.metrics is not None:
        #no need to load the coco2014 validation dataset if no evaluation is needed during distillation
        c.dataset_kwargs = dnnlib.EasyDict(class_name='training.mscoco_dataset.ImageDataset', path=opts.data, resolution=opts.resolution, random_flip=opts.xflip)    
        print(c.dataset_kwargs)
    
    # if (opts.metrics is not None) or (not opts.train_mode):
    #     #no need to load the coco2014 validation dataset if no evaluation is needed during distillation
    #     c.dataset_kwargs = dnnlib.EasyDict(class_name='training.mscoco_dataset.ImageDataset', path=opts.data, resolution=opts.resolution, random_flip=opts.xflip)    
    #     print(c.dataset_kwargs)
    # else:
    #     c.dataset_kwargs = c.dataset_prompt_text_kwargs 
    
    
    
    # Validate dataset options.
    try:
        if opts.train_mode:
            dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_prompt_text_kwargs)
        else:
            dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        data_max_size = len(dataset_obj) # be explicit about dataset size
    except IOError as err:
        raise click.ClickException(f'--data: {err}')


    c.network_kwargs.update(use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)
    c.update(loss_scaling_G=opts.lsg, cudnn_benchmark=opts.bench)
    
    c.alpha = opts.alpha
    c.tmax = opts.tmax
    c.tmin = opts.tmin

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    if opts.resume is not None:
        c.resume_training = opts.resume
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_kimg = int(match.group(1))

    # Description string.
    cond_str = 'text_cond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-glr{opts.glr}-lr{opts.lr}-initsigma{opts.init_timestep}-gpus{dist.get_world_size():d}-alpha{c.alpha}-batch{c.batch_size:d}-tmax{c.tmax:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    
    print(opts.outdir)
    print(opts.data)
    
    
    
    

    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    c.metric_pt_path = opts.metric_pt_path
    c.metric_open_clip_path = opts.metric_open_clip_path
    c.metric_clip_path = opts.metric_clip_path


    c.fake_score_use_lora = opts.fake_score_use_lora    
    c.pretrained_model_name_or_path = opts.sd_model
    c.pretrained_vae_model_name_or_path = opts.sd_model

    c.cfg_train_fake = opts.cfg_train_fake
    c.cfg_eval_fake = opts.cfg_eval_fake
    c.cfg_eval_real = opts.cfg_eval_real
    c.num_steps = opts.num_steps
    
    c.train_mode = opts.train_mode
    c.network_pkl = opts.network_pkl
    
    c.enable_xformers = opts.enable_xformers
    c.gradient_checkpointing = opts.gradient_checkpointing
    

    

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    #dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Dataset length:          {data_max_size}')
    dist.print0(f'Class-conditional:       text_cond')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'alpha:                   {c.alpha}')
    dist.print0(f'tmax:                    {c.tmax}')
    dist.print0(f'tmin:                    {c.tmin}')
    dist.print0(f'precision:               {dtype_str}')
    dist.print0(f'metric_pt_path:          {c.metric_pt_path}')
    dist.print0(f'pretrained_model_name_or_path: {c.pretrained_model_name_or_path}')
    dist.print0(f'pretrained_vae_model_name_or_path: {c.pretrained_vae_model_name_or_path}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)


    
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
