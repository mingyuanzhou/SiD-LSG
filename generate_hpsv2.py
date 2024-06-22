# Modified from the EDM generate.py file
# Copyright (c) 2022, Huangjie Zheng. All rights reserved.
#
# This work is licensed under a MIT License.


import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
import hpsv2
from functools import partial
from training.sid_sd_util import load_sd15, sid_sd_sampler #, swiftbrush_sd_sampler

from diffusers import DiffusionPipeline
#from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDPMScheduler
#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

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

#----------------------------------------------------------------------------

def compress_to_npz(folder_path, num=50000):
    # Get the list of all files in the folder
    npz_path = f"{folder_path}.npz"
    file_names = os.listdir(folder_path)

    # Filter the list of files to include only images
    file_names = [file_name for file_name in file_names if file_name.endswith(('.png', '.jpg', '.jpeg'))]
    num = min(num, len(file_names))
    file_names = file_names[:num]

    # Initialize a dictionary to hold image arrays and their filenames
    samples = []

    # Iterate through the files, load each image, and add it to the dictionary with a progress bar
    for file_name in tqdm.tqdm(file_names, desc=f"Compressing images to {npz_path}"):
        # Create the full path to the image file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the image using PIL and convert it to a NumPy array
        image = PIL.Image.open(file_path)
        image_array = np.asarray(image).astype(np.uint8)
        
        samples.append(image_array)
    samples = np.stack(samples)

    # Save the images as a .npz file
    np.savez(npz_path, arr_0=samples)
    print(f"Images from folder {folder_path} have been saved as {npz_path}")

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-799', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--num', 'num_fid_samples',  help='Maximum num of images', metavar='INT',                             type=click.IntRange(min=1), default=50000, show_default=True)
@click.option('--init_timestep', 'init_timestep',      help='Stoch. noise std', metavar='long',                      type=int, default=625, show_default=True)
@click.option('--repo_id', 'repo_id',   help='diffusion pipeline filename', metavar='PATH|URL',         type=str, default='runwayml/stable-diffusion-v1-5', show_default=True)


def main(network_pkl, outdir, subdirs, seeds, max_batch_size, num_fid_samples, init_timestep, coco14_captions30k,repo_id,device=torch.device('cuda')):
    """Generate random images using SiD-LSG".

    Examples:

    \b
    # Generate images with hpsv2 prompts and evaluate them
    torchrun --standalone --nproc_per_node=<num gpu> generate_hpsv2.py --outdir=out --batch=16 --network=<network_path> \\
        --repo_id=<SD 1.5 or 2.1 depending on --network>
        
    
    python generate_hpsv2.py --outdir='image_experiment/sd2.1_example_images'  --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base' 
    
    #--text_prompts='prompts/fig6-prompts.txt' --seeds=0
        
    """
    dist.init()
    dtype=torch.float16

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts('all') 

    
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        G_ema = pickle.load(f)['ema'].to(device).to(dtype)

    pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)

    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device, dtype)
    vae = pipeline.vae.to(device, dtype)
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    num_steps = 1    
    num_steps_eval = 1
    G=partial(sid_sd_sampler,unet=G_ema,noise_scheduler=noise_scheduler,
                                                    text_encoder=text_encoder, tokenizer=tokenizer, 
                                                    resolution=512,dtype=dtype,return_images=True,vae=vae,num_steps=num_steps,train_sampler=False,num_steps_eval=num_steps_eval)   

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    for key in all_prompts:
        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{outdir}/{key}"...')
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            img_channels = 4
            img_resolution = 64
            latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)

            c=[all_prompts[key][i] for i in batch_seeds]
            
            with torch.no_grad():
                images = G(latents=latents, contexts=c, init_timesteps=init_timestep * torch.ones((len(c),), device = latents.device, dtype=torch.long))
            
            # Save images.
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, key, f'{seed-seed%1000:06d}') if subdirs else os.path.join(outdir, key)
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:05d}.jpg')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)

    # Done.
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        hpsv2.evaluate(outdir, hps_version="v2.0")
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
