# Copyright (c) 2024, Mingyuan Zhou. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


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
from functools import partial
from training.sid_sd_util import load_sd15, sid_sd_sampler

import torch
from diffusers import DiffusionPipeline
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

def read_file_to_sentences(filename):
    # Initialize an empty list to store the sentences
    sentences = []

    # Open the file
    with open(filename, 'r', encoding='utf-8') as file:
        # Read each line from the file
        for line in file:
            # Strip newline and any trailing whitespace characters
            clean_line = line.strip()
            # Add the cleaned line to the list if it is not empty
            if clean_line:
                sentences.append(clean_line)
    
    return sentences

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
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--num', 'num_fid_samples',  help='Maximum num of images', metavar='INT',                             type=click.IntRange(min=1), default=30000, show_default=True)
@click.option('--init_timestep', 'init_timestep',      help='t_init, in [0,999]', metavar='INT',                    type=click.IntRange(min=0), default=625, show_default=True)
@click.option('--text_prompts', 'text_prompts',   help='captions filename; the default [prompts/captions.txt] consists of 30k COCO2014 prompts', metavar='PATH|URL',         type=str, default='prompts/captions.txt', show_default=True)
@click.option('--repo_id', 'repo_id',   help='diffusion pipeline filename', metavar='PATH|URL',                     type=str, default='runwayml/stable-diffusion-v1-5', show_default=True)
@click.option('--use_fp16',             help='Enable mixed-precision training', metavar='BOOL',                     type=bool, default=True, show_default=True)
@click.option('--enable_compress_npz',         help='Enable compressinve npz', metavar='BOOL',                             type=bool, default=False, show_default=True)
@click.option('--num_steps_eval', 'num_steps_eval',      help='Set as 1 by default, but the code support >1', metavar='INT',      type=click.IntRange(min=0), default=1, show_default=True)
@click.option('--custom_seed',             help='Enable custom seed', metavar='BOOL',                     type=bool, default=False, show_default=True)



def main(network_pkl, outdir, subdirs, seeds, max_batch_size, num_fid_samples, init_timestep, text_prompts,repo_id,device=torch.device('cuda'),use_fp16=True,enable_compress_npz=False,num_steps_eval=1,custom_seed=False):
    """Generate random images using SiD-LSG. fp16 is used by default for evaluation.

    #The model checkpoints are available at https://huggingface.co/UT-Austin-PML/SiD-LSG

    #Download and place them into '/data/Austin-PML/SiD-LSG/' or a folder you choose, and then run the following to generate example images or 30k images to compute various metrics
    #When running the following examples, you can also replace '/data/Austin-PML/SiD-LSG/' with 'https://huggingface.co/UT-Austin-PML/SiD-LSG/resolve/main/'  to directly download the checkpoint from HuggingFace


    Examples:
    
    #Reproduce Figure 1:
    
    python generate_onestep.py --outdir='image_experiment/example_images/figure1' --seeds='8,8,2,3,2,1,2,4,3,4' --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'  --text_prompts='prompts/fig1-captions.txt'  --custom_seed=1


    #Reproduce Figure 6 (the columns labeled SD1.5 and SD2.1), ensuring the seeds align with the positions of the prompts within the HPSV2 defined list of prompts:

    python generate_onestep.py --outdir='image_experiment/example_images/figure6/sd1.5' --seeds='668,329,291,288,057,165' --batch=6 --network='/data/Austin-PML/SiD-LSG/batch512_cfg4.54.54.5_t625_8380_v2.pkl' --text_prompts='prompts/fig6-captions.txt' --custom_seed=1

    python generate_onestep.py --outdir='image_experiment/example_images/figure6/sd2.1base' --seeds='668,329,291,288,057,165' --batch=6 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'  --text_prompts='prompts/fig6-captions.txt' --custom_seed=1


    #Reproduce Figure 8:

    python generate_onestep.py --outdir='image_experiment/example_images/figure8' --seeds='4,4,1,1,4,4,1,1,2,7,7,6,1,20,41,48' --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'  --text_prompts='prompts/fig8-captions.txt' --custom_seed=1

        
        
        
        
    # Generate 30000 images using 4 GPUs, which are then used to compute FID, CLIP, HPSv2, Precsion, and Recall, as shown in Tables 1 and 2 of the SiD-LSG paper: https://arxiv.org/abs/2406.01561   

    #Stable Diffusion 1.5

        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 1.5
        #FID 8.71, CLIP 0.302
        #run:
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd1.5_kappa1.5/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_cfg1.51.51.5_t625_8806_v2.pkl'  
        #or run:
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd1.5_kappa1.5/fake_images' --seeds=0-29999 --batch=16 --network='https://huggingface.co/UT-Austin-PML/SiD-LSG/resolve/main/batch512_cfg1.51.51.5_t625_8806_v2.pkl'


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 1.5, longer training
        #FID 8.15, CLIP 0.304     
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd1.5_kappa1.5_traininglonger/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_cfg1.51.51.5_t625_18789_v2.pkl'  


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 2 
        #FID 9.56, CLIP 0.313    
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd1.5_kappa2/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_cfg222_t625_4812_v2.pkl' 


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 3 
        #FID 13.21, CLIP 0.314   
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd1.5_kappa3/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_cfg333_t625_6349_v2.pkl' 


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 4.5 
        #FID 16.59, CLIP 0.317  
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd1.5_kappa4.5/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_cfg4.54.54.5_t625_8380_v2.pkl' 



    #Stable Diffusion 2.1-base

        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 1.5
        #FID 9.52, CLIP 0.308     
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd2.1base_kappa1.5/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg1.51.51.5_t625_9728_v2.pkl'  --repo_id='stabilityai/stable-diffusion-2-1-base'


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 2 
        #FID 10.97, CLIP 0.318    
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd2.1base_kappa2/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg222_t625_8482_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 3 
        #FID 13.50, CLIP 0.321   
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd2.1base_kappa3/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg333_t625_6144_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'


        #SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 4.5 
        #FID 16.54, CLIP 0.322  
        torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sd2.1base_kappa4.5/fake_images' --seeds=0-29999 --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'
        
     

        
    """
    dist.init()
    
    dtype=torch.float16 if use_fp16 else torch.float32
        
    captions = read_file_to_sentences(text_prompts)
    
    
        

    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    if not custom_seed:
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    else:
        seeds_idx = parse_int_list(f'0-{len(seeds)-1}')
        all_batches = torch.as_tensor(seeds_idx).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    
    
    #print(rank_batches)
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if 1:    
        # Evaluate sid-lsg
        # Load network.
        dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
            G_ema = pickle.load(f)['ema'].to(device).to(dtype)

        pipeline = DiffusionPipeline.from_pretrained(repo_id, use_safetensors=True)

        # 1. Load the tokenizer and text encoder to tokenize and encode the text. 
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder.to(device, dtype)

        # 2. Load the autoencoder model which will be used to decode the latents into image space. 
        vae = pipeline.vae.to(device, dtype)
        noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        
        del pipeline
        
        num_steps = 1    
        
        G=partial(sid_sd_sampler,unet=G_ema,noise_scheduler=noise_scheduler,
                                                     text_encoder=text_encoder, tokenizer=tokenizer, 
                                                     resolution=512,dtype=dtype,return_images=True,vae=vae,num_steps=num_steps,train_sampler=False,num_steps_eval=num_steps_eval)     

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()

        if num_steps_eval>1:
            outdir = f'{outdir}_numstep{num_steps_eval}'    
            
        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue
            # Pick latents and labels.
            
            if not custom_seed:
                rnd = StackedRandomGenerator(device, batch_seeds)
            else:
                cseed= [seeds[i] for i in batch_seeds]
                rnd = StackedRandomGenerator(device, cseed)
            
            img_channels=4
            img_resolution=64
            latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)

            c = [captions[i] for i in batch_seeds]  # Index captions using list comprehension

            with torch.no_grad():
                images = G(latents=latents, contexts=c, init_timesteps=init_timestep * torch.ones((len(c),), device = latents.device, dtype=torch.long))

            # Save images.
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            del images
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)
            del images_np
        # Done.
        
    else:
        #for sd-turbo and sdxl-turbo
        from diffusers import AutoPipelineForText2Image
      
        if 0:
            #sd-turbo
            pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo") #, torch_dtype=torch.float16, variant="fp16")
        else:
            #sdxl-turbo
            pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo") 
        pipe.to(device)

        if dist.get_rank() == 0:
            torch.distributed.barrier()

        # Loop over batches.
        dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            img_channels=4
            img_resolution=64
            latents = rnd.randn([batch_size, img_channels, img_resolution, img_resolution], device=device)

            c = [captions[i] for i in batch_seeds]  # Index captions using list comprehension

            with torch.no_grad():
                images = pipe(prompt=c, num_inference_steps=1, guidance_scale=0.0).images
            images_np = [np.array(image) for image in images]    
            # Save images.
            #images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            del images
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)
            del images_np
        # Done.
    
    if enable_compress_npz:
    
        torch.distributed.barrier()
        if dist.get_rank() == 0:
            compress_to_npz(outdir, num_fid_samples)
        torch.distributed.barrier()
    dist.print0('Done.')
    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
