import os
import numpy as np
import scipy.linalg
from . import sid_metric_utils as metric_utils


from functools import partial

import dnnlib



#----------------------------------------------------------------------------

def compute_fid_and_clip(opts, max_real, num_gen,batch_size=64,compute_clip=False):
    
    
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'

    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real,batch_size=batch_size).get_mean_cov()


    open_clip_detector_url=opts.metric_open_clip_path


    if not compute_clip:
        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen,batch_size=batch_size).get_mean_cov()
        open_clip_score = float('nan')
        clip_score = float('nan')
    else:
        clip_score_fn = opts.clip_score_fn
        stats,open_clip_score, clip_score = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen,batch_size=batch_size,compute_clip=compute_clip, clip_score_fn=clip_score_fn,open_clip_detector_url=open_clip_detector_url)
        mu_gen, sigma_gen = stats.get_mean_cov()

    if opts.rank != 0:
        if not compute_clip:
            return float('nan')
        else:
            return float('nan'),float('nan'),float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    if not compute_clip:
        return float(fid)
    else:
        return float(fid), float(open_clip_score), float(clip_score) 

#----------------------------------------------------------------------------
