import os
import numpy as np
import scipy.linalg
from . import sid_metric_utils as metric_utils

import tqdm
from functools import partial
from torch_utils import distributed as dist
import dnnlib



#----------------------------------------------------------------------------

def compute_hpsv2(opts, num_gen=800, batch_size=64):
    # Setup generator and load labels.
    G = opts.G
    init_timestep = opts.init_timestep
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all') 
    min_num_prompts = min([len(all_prompts[key]) for key in all_prompts])
    num_gen = min(num_gen, min_num_prompts)

    num_batches = ((len(num_gen) - 1) // (batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(num_gen).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]


    outdir = dnnlib.make_cache_dir_path('hpsv2')

    # Image generation func.
    def run_generator(z, c):
        with torch.no_grad():
            img = G(latents=z, contexts=c, init_timesteps=opts.init_timestep * torch.ones((len(c),), device=opts.device, dtype=torch.long))

        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img

    for key in all_prompts:
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            bs = len(batch_seeds)
            if bs == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            img_channels = 4
            img_resolution = 64
            latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)
            contexts = [all_prompts[key][i] for i in batch_seeds]
            images = run_generator(z, contexts)

            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, key)
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:05d}.jpg')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    torch.distributed.barrier()
    if dist.get_rank() == 0:
        hpsv2.evaluate(outdir, hps_version="v2.0")
    torch.distributed.barrier()

    return

#----------------------------------------------------------------------------
