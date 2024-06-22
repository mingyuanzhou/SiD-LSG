# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#Mingyuan Zhou, added the computation of clip score

import os
import time
import json
import torch
import dnnlib

from . import sid_metric_utils as metric_utils
#from . import sid_frechet_inception_distance as frechet_inception_distance
from . import sid_fid_and_clip as fid_and_clip
from . import sid_hpsv2 as hpsv2
#from . import sid_kernel_inception_distance
#from . import sid_precision_recall as precision_recall
#from . import perceptual_path_length
#from . import sid_inception_score as inception_score



#----------------------------------------------------------------------------

_metric_dict = dict() # name => fn

def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

#----------------------------------------------------------------------------

def calc_metric(metric, **kwargs): # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results         = dnnlib.EasyDict(results),
        metric          = metric,
        total_time      = total_time,
        total_time_str  = dnnlib.util.format_time(total_time),
        num_gpus        = opts.num_gpus,
    )

#----------------------------------------------------------------------------


def append_line(jsonl_line, fname):
    with open(fname, 'at') as f:
        f.write(jsonl_line + '\n')


def report_metric(result_dict, run_dir=None, snapshot_pkl=None, alpha=None,num_steps_eval=None):
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        if alpha is None:
            append_line(jsonl_line=jsonl_line, fname=os.path.join(run_dir, f'metric-{metric}.jsonl'))
        else:
            if num_steps_eval is None or num_steps_eval==1:
                append_line(jsonl_line=jsonl_line, fname=os.path.join(run_dir, f'metric-{metric}-alpha-{alpha:03f}.jsonl'))
            else:
                append_line(jsonl_line=jsonl_line, fname=os.path.join(run_dir, f'metric-{metric}-alpha-{alpha:03f}-num_steps_eval-{num_steps_eval:02d}.jsonl'))

#----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid30k_full(opts):
    fid = fid_and_clip.compute_fid_and_clip(opts, max_real=None, num_gen=30000,batch_size=8,compute_clip=False)
    return dict(fid30k_full=fid,open_clipscore_30k=float('nan'),clipscore30k=float('nan'))

@register_metric
def fid_clip_30k_full(opts):
    fid,open_clip_score,clip_score = fid_and_clip.compute_fid_and_clip(opts, max_real=None, num_gen=30000,batch_size=8,compute_clip=True)
    return dict(fid30k_full=fid,open_clipscore_30k=open_clip_score,clipscore30k=clip_score)

@register_metric
def fid_test(opts):
    fid = fid_and_clip.compute_fid_and_clip(opts, max_real=None, num_gen=1, batch_size=1,compute_clip=False)
    
    return dict(fid30k_full=fid,open_clipscore_30k=float('nan'),clipscore30k=float('nan'))

@register_metric
def fid_clip_test(opts):
    fid,open_clip_score,clip_score = fid_and_clip.compute_fid_and_clip(opts, max_real=None, num_gen=1, batch_size=1,compute_clip=True)
        
    return dict(fid30k_full=fid,open_clipscore_30k=open_clip_score,clipscore30k=clip_score)

@register_metric
def hpsv2(opts):
    hpsv2.compute_hpsv2(opts, num_gen=800, batch_size=64)