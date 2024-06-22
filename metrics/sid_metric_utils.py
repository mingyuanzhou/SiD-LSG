# Copyright (c) 2024, Mingyuan Zhou.  All rights reserved. Modified from metric_utils.py




# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import dnnlib

from typing import Iterator, Callable, Optional, Tuple, Union
import functools


from functools import partial
import dill

import dnnlib
from pathlib import Path
from networks.clip import CLIP

def build_clip(url: str) -> None:
    clip = CLIP('ViT-g-14', pretrained='laion2b_s12b_b42k')
    Path(url).parent.mkdir(parents=True, exist_ok=True)
    with open(url, 'wb') as f:
        dill.dump(clip, f)


# Function to safely load the file
def safe_load(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = dill.load(file)
        return model
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return None


#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self,metric_pt_path=None,metric_open_clip_path=None,metric_clip_path=None, G=None, init_timestep=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, local_rank=0, device=None, progress=None, cache=True):

        

        assert 0 <= rank < num_gpus
        
        
        self.metric_clip_path = metric_clip_path
    
        self.clip_score_fn = None #Specify this to use a user-provided CLIP encoder to calculate the clip score; not used by default
        
        
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.init_timestep = init_timestep
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.local_rank     = local_rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        

        cache_dir = dnnlib.make_cache_dir_path('detectors')
        open_clip_detector_url = os.path.join(cache_dir, 'clipvitg14.pkl')
        detector_kwargs = {'texts': None, 'div255': True}
        if rank == 0:
            print(f"detector_url: {open_clip_detector_url}")

        # If it does not exist, build and save CLIP.
        if not os.path.exists(open_clip_detector_url) and rank == 0:
            build_clip(open_clip_detector_url)

        self.metric_open_clip_path = open_clip_detector_url
#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]


class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    ##added For COCO_val256 FID computation
    #opts.dataset_kwargs.resolution=256
    ###
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.local_rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats



from PIL import Image
import torchvision.transforms.functional as TF



import PIL.Image

def make_transform(
    transform: Optional[str],
    output_width: Optional[int],
    output_height: Optional[int]
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    def scale(width, height, img):
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop(width, height, img):
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        return np.array(img)

    def center_crop_wide(width, height, img):
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        if img.ndim == 2:
            img = img[:, :, np.newaxis].repeat(3, axis=2)
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), PIL.Image.Resampling.LANCZOS)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + 'transform')
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if output_width is None or output_height is None:
            raise click.ClickException('must specify --resolution=WxH when using ' + transform + ' transform')
        return functools.partial(center_crop_wide, output_width, output_height)
    assert False, 'unknown transform'


def resize_images_in_tensor(input_tensor, new_size=(256, 256)):
    resized_images = []

    # Process each image in the batch
    for i in range(input_tensor.size(0)):
        # Convert the PyTorch tensor to a PIL Image
        pil_image = TF.to_pil_image(input_tensor[i])

        # Resize the image
        resized_pil_image = pil_image.resize(new_size, Image.LANCZOS)

        # Convert back to a PyTorch tensor
        resized_tensor = TF.to_tensor(resized_pil_image).to(device=input_tensor.device)

        # Normalize the tensor to the original range and data type
        resized_tensor = resized_tensor * 255
        resized_tensor = resized_tensor.byte()

        resized_images.append(resized_tensor)

    # Stack all the resized images together to get back the original batch shape
    output_tensor = torch.stack(resized_images, dim=0)
    return output_tensor


import torchvision.transforms as transforms

def resize_image(image, new_size=(256, 256)):
    """
    Resize an image to the specified size using PyTorch transforms.

    Args:
    - image_path (str): Path to the input image file.
    - new_size (tuple): Tuple specifying the new size, e.g., (width, height).

    Returns:
    - resized_image (Tensor): Resized image as a PyTorch tensor.
    """

    # Load the image
    #image = Image.open(image_path)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(new_size),
        transforms.ToTensor()  # Convert the image to a PyTorch tensor
    ])

    # Apply the transformation
    resized_image = transform(image)

    return resized_image



#----------------------------------------------------------------------------

from torch_utils import misc

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, jit=False, compute_clip =False,clip_score_fn=None,open_clip_detector_url=None,**stats_kwargs):

    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and load labels.
    G = opts.G
    init_timestep = opts.init_timestep
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    dataset_sampler = misc.InfiniteSampler(dataset=dataset, rank=opts.rank, num_replicas=opts.num_gpus, seed=0)
    dataloader = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=dataset_sampler, batch_size=batch_gen))

    
    # Image generation func.
    def run_generator(z, c):
        with torch.no_grad():
            img = G(latents=z, contexts=c, init_timesteps=opts.init_timestep * torch.ones((len(c),), device=opts.device, dtype=torch.long))

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    # JIT.
    # if jit:
    #     z = init_timestep*torch.zeros([batch_gen, G.img_channels, G.img_resolution, G.img_resolution], device=opts.device)
    #     c = torch.zeros([batch_gen, G.c_dim], device=opts.device)
    #     run_generator = torch.jit.trace(run_generator, [z, c, init_timestep], check_trace=False)

    # Initialize.
    stats = FeatureStats(**stats_kwargs)

    #stats_kwargs1 = stats_kwargs
    #clip_stats = FeatureStats(**stats_kwargs)

    
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    
    transform_image = make_transform(transform=None,output_width=256,output_height=256)
    # Main loop.

    if compute_clip:
        open_clip_detector =safe_load(open_clip_detector_url).to(opts.device)
        
    
    clip_score=[]
    open_clip_score=[]
    while not stats.is_full():
        images = []
        texts = []
        for _i in range(batch_size // batch_gen):
            #print(_i)
            _, contexts = next(dataloader)
            z = torch.randn([batch_gen, 4, dataset.resolution//8, dataset.resolution//8], device=opts.device)
            images.append(run_generator(z, contexts))
            texts.extend(contexts)
           
            #print(texts)
        images = torch.cat(images)
        
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        if 1:    
            images = resize_images_in_tensor(images)
        else:
            images_np = images.permute(0, 2, 3, 1).cpu().numpy()
            resized_images = []
            # Process each image in the batch
            for image in images_np:
                img_256=transform_image(image)
                img_256_tensor=torch.tensor(img_256, dtype=torch.uint8,device=opts.device)
                resized_images.append(img_256_tensor)
            images = torch.stack(resized_images, dim=0).permute(0, 3,1,2)
        
        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)

        if compute_clip:

            if clip_score_fn is not None:
                with torch.no_grad():
                    clip_score_i = clip_score_fn(images, texts).cpu().numpy() 
            else:
                clip_score_i = 'nan' 
            clip_score.append(clip_score_i)
            
            open_clip_detector_kwargs = {'texts': texts, 'div255': True}
            
            clip_features = open_clip_detector(images, **open_clip_detector_kwargs)
            image_features, text_features = clip_features.tensor_split((clip_features.size(1)//2,), 1)
            open_clip_score.append(float((image_features * text_features).sum(-1).mean()))

        progress.update(stats.num_items)
    if not compute_clip:
        return stats
    else:
        return stats, np.array(open_clip_score).mean(),np.array(clip_score).mean() #.cpu().numpy()

#----------------------------------------------------------------------------
