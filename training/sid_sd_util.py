# Copyright (c) 2024, Mingyuan Zhou and Zhendong Wang. All rights reserved.
#
# This work is licensed under APACHE LICENSE, VERSION 2.0
# You should have received a copy of the license along with this
# work. If not, see https://www.apache.org/licenses/LICENSE-2.0.txt


import torch
import diffusers
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from packaging import version
from diffusers.loaders import StableDiffusionXLLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available

from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    XFormersAttnProcessor,
    LoRAXFormersAttnProcessor,
    LoRAAttnProcessor2_0,
    FusedAttnProcessor2_0,
)
def upcast_vae(vae):
    dtype = vae.dtype
    vae.to(dtype=torch.float32)
    use_torch_2_0_or_xformers = isinstance(
        vae.decoder.mid_block.attentions[0].processor,
        (
            AttnProcessor2_0,
            XFormersAttnProcessor,
            LoRAXFormersAttnProcessor,
            LoRAAttnProcessor2_0,
            FusedAttnProcessor2_0,
        ),
    )
    # if xformers or torch_2_0 is used attention block does not need
    # to be in float32 which can save lots of memory
    if use_torch_2_0_or_xformers:
        vae.post_quant_conv.to(dtype)
        vae.decoder.conv_in.to(dtype)
        vae.decoder.mid_block.to(dtype)


def load_sd15(pretrained_model_name_or_path, pretrained_vae_model_name_or_path, device, weight_dtype, 
              revision=None, variant=None, lora_config=None, enable_xformers=False, gradient_checkpointing=False):
    # Load the tokenizer
    print(f'pretrained_model_name_or_path: {pretrained_model_name_or_path}')
    print(f'revision: {revision}')

    #print('tokenizer start')
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    if noise_scheduler.config.prediction_type == "v_prediction":
        noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)
    
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path,subfolder="text_encoder", revision=revision, variant=variant
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, variant=variant
    )

    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision, variant=variant
    )

    # Freeze untrained components
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Move unet and text_encoders to device and cast to weight_dtype
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)

    # # Add LoRA setting if needed
    # if lora_config is not None:
    #     unet.requires_grad_(False)
    #     # Add adapter and make sure the trainable params are in float32.
    #     unet.add_adapter(lora_config)
    #     #if weight_dtype == "fp16":
    #     if weight_dtype == torch.float16:
    #         for param in unet.parameters():
    #             # only upcast trainable parameters (LoRA) into fp32
    #             if param.requires_grad:
    #                 param.data = param.to(torch.float32)

    if enable_xformers:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                ValueError(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    return unet, vae, noise_scheduler, text_encoder, tokenizer


def tokenize_captions(tokenizer, examples):
    max_length = tokenizer.model_max_length
    captions = []
    for caption in examples:
        captions.append(caption)

    text_inputs = tokenizer(
        captions, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    return text_inputs.input_ids


@torch.no_grad()
def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = None

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    print(model_class)
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

                    
def sid_sd_sampler(unet, latents, contexts,init_timesteps,  noise_scheduler, 
                         text_encoder, tokenizer, resolution, dtype=torch.float16,return_images=False, vae=None,guidance_scale=1,num_steps=1,train_sampler=True,num_steps_eval=1):
    #The single step version (num_steps=num_steps_eval=1) has been fully tested; the multi-step version is working in progress
    
    # Get the text embedding for conditioning
    prompt=contexts
    batch_size = len(prompt)
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(latents.device))[0]
    #print('Finished creating text embedding.')
    
    
    if train_sampler:
        D_x = torch.zeros_like(latents).to(latents.device)
        step_indices = [torch.tensor(0).to(latents.device)]  # Initial step
        for i in range(num_steps):
            noise = latents if i == 0 else torch.randn_like(latents).to(latents.device)
            init_timesteps_i = (init_timesteps*(1-i/num_steps)).to(torch.long)
            latents = noise_scheduler.add_noise(D_x, noise, init_timesteps_i).to(torch.float32)
            latent_model_input = noise_scheduler.scale_model_input(latents, init_timesteps_i) 
            noise_pred = unet(latent_model_input.to(dtype), init_timesteps_i, encoder_hidden_states=text_embeddings).sample.to(torch.float32)
            D_x  = noise_scheduler.step(noise_pred, init_timesteps_i[0], latents,return_dict=True).pred_original_sample.to(torch.float32)  
    else:
        D_x = torch.zeros_like(latents).to(latents.device)
        for i in range(num_steps_eval):
            noise = latents if i == 0 else torch.randn_like(latents).to(latents.device)
            init_timesteps_i = (init_timesteps*(1-i/num_steps_eval)).to(torch.long)
            latents = noise_scheduler.add_noise(D_x, noise, init_timesteps_i)
            latent_model_input = noise_scheduler.scale_model_input(latents, init_timesteps_i) 
            with torch.no_grad():
                noise_pred = unet(latent_model_input.to(dtype), init_timesteps_i, encoder_hidden_states=text_embeddings).sample
            D_x  = noise_scheduler.step(noise_pred, init_timesteps_i[0], latents,return_dict=True).pred_original_sample
        D_x = D_x.to(torch.float32)  

    if return_images:
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = vae.dtype == torch.float16 and vae.config.force_upcast
        if needs_upcasting:
            upcast_vae(vae=vae)
            D_x = D_x.to(next(iter(vae.post_quant_conv.parameters())).dtype)
        images = vae.decode(D_x / vae.config.scaling_factor, return_dict=False)[0]
        #images = vae.decode(D_x /0.18215).sample
        # cast back to fp16 if needed
        if needs_upcasting:
            vae.to(dtype=torch.float16)
        return images.to(torch.float32)
    else:
        return D_x.to(torch.float32)
    
                   
def sid_sd_denoise(unet, images, noise, contexts,timesteps,  noise_scheduler, 
                         text_encoder, tokenizer, resolution,dtype=torch.float16,predict_x0=True,guidance_scale=1):
    # Get the text embedding for conditioning
    
    prompt=contexts
    batch_size = len(prompt)

    text_input = tokenizer(
        prompt,
        padding='max_length',
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt',
    )
    text_embeddings = None
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(images.device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [''] * batch_size,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(images.device))[0]
    
    latents = noise_scheduler.add_noise(images, noise, timesteps)
    if guidance_scale==1:
        latent_model_input = noise_scheduler.scale_model_input(latents, timesteps) 
        noise_pred = unet(latent_model_input.to(dtype), timesteps, encoder_hidden_states=text_embeddings).sample.to(torch.float32)
    else:
        if 0:
            # Convert latent_model_input to dtype
            latent_model_input = noise_scheduler.scale_model_input(latents, timesteps).to(dtype)
            # Compute noise_pred
            noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states=text_embeddings).sample.to(torch.float32)
            #noise_pred =noise_pred.to(torch.float32)
            # Compute noise_pred_uncond
            noise_pred_uncond = unet(latent_model_input, timesteps, encoder_hidden_states=uncond_embeddings).sample.to(torch.float32)
            #noise_pred_uncond =noise_pred_uncond.to(torch.float32)
            # Compute guided noise_pred
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
        else:
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            t = torch.cat([timesteps, timesteps])
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred = unet(latent_model_input.to(dtype), t, encoder_hidden_states=text_embeddings).sample.to(torch.float32)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        
    if predict_x0:
        latents = latents.to(torch.float32)
        D_x = [noise_scheduler.step(n, t, z,return_dict=True).pred_original_sample.to(torch.float32) for n, t, z in zip(noise_pred, timesteps, latents)]
        D_x = torch.stack(D_x).to(torch.float32)
        return D_x
    else:
        return noise_pred.to(torch.float32)
        

if __name__ == "__main__":
    import os
    os.environ['HF_HOME'] = '/blabla/cache/'

    device = 'cuda:0'
    weight_dtype = torch.float16
    # from peft import LoraConfig
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=8,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )

    pretrained_model_name_or_path = 'runwayml/stable-diffusion-v1-5'
    pretrained_vae_model_name_or_path = None
    
    unet, vae, noise_scheduler, text_encoder, tokenizer = load_sd15(pretrained_model_name_or_path=pretrained_model_name_or_path, pretrained_vae_model_name_or_path=None,
                                                                    device=device, weight_dtype=weight_dtype, enable_xformers=True, lora_config=None)

    examples = ['a cute corgi running on a grass', 'a cute cat sitting on a sofar']
    tokens = tokenize_captions(tokenizer, examples)

    # Get the text embedding for conditioning
    prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoder, tokens)
    print(prompt_embeds.shape)
