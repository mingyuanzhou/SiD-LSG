#!/bin/bash

model=$1

# Uncomment to set CUDA visible devices and run specific script

# export CUDA_VISIBLE_DEVICES=0,1,2,3

#sh run_sid.sh 'sd1.5' 
#sh run_sid.sh 'sd2.1_base' 

#sh run_sid.sh 'sd1.5_fp16' 
#sh run_sid.sh 'sd2.1_base_fp16'

# Modify --duration to reproduce the reported results



# kappa1 = cfg_train_fake
# kappa2=kappa3 = cfg_eval_fake
# kappa4 = cfg_eval_real

# To resume from a saved checkpoint, add the following option:

# --resume 'image_experiment/sid-lsg-train-runs/*.pt' 

# If your system has sufficient memory, you can choose to evaluate both FID and CLIP scores during training by adding:

# --metrics 'fid_clip_30k_full'

# otherwise, you can compute FID only:

# --metrics 'fid30k_full'

# or don't specify `metrics` so it will take the default value of None.


# To reproduce the results reported in the paper, choose `--enable_xformers 0' and then run
#     sh run_sid.sh 'sd1.5'    
#     sh run_sid.sh 'sd2.1-base'
    
# As mentioned in the paper, our computing infrastracture used for SiD-LSG does not support xformers but it is highly recommeded to enable it if your computing platform allows it; see `https://huggingface.co/docs/diffusers/v0.21.0/en/optimization/xformers` for more details

# If memory constraints are a concern, consider training the model using 16-bit floating point precision (fp16):
#     sh run_sid.sh 'sd1.5-fp16' 
#     sh run_sid.sh 'sd2.1-base-fp16'   

# While this approach significantly reduces memory and is also found to lead to faster convergence, it will require gradient clipping to prevent suddern divergence and result in inferior model performance. See Figure 7 of https://arxiv.org/abs/2406.01561 for more details. 

# You can save memory by disabling the Exponential Moving Average (EMA) feature. To do this, add:

# --ema 0

# You can also consider enableing gradient_checkpointing to save memory:

# --gradient_checkpointing 1




if [ "$model" = 'sd1.5' ]; then
    # Command to run torch with specific parameters
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
    --duration 10 \
    --enable_xformers 1 \
    --gradient_checkpointing 0 \
    --metrics 'fid30k_full' \
    --ema 0.05
    #--nosubdir  \
    #--metrics 'fid_30k_full' \
    #--metrics 'fid_clip_30k_full' \
    #--ema 0.05
    
elif [ "$model" = 'sd2.1_base' ]; then
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
    --sd_model "stabilityai/stable-diffusion-2-1-base" \
    --tick 2 \
    --snap 50 \
    --dump 100 \
    --lr 0.000001 \
    --glr 0.000001 \
    --duration 10 \
    --enable_xformers 1 \
    --gradient_checkpointing 0 \
    --ema 0.05
    #--nosubdir  \
    #--metrics 'fid_30k_full' \
    #--metrics 'fid_clip_30k_full' \
    #--ema 0.05

elif [ "$model" = 'sd1.5_fp16' ]; then
    # Command to run torch with specific parameters
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
    --fp16 1 \
    --batch-gpu 1 \
    --sd_model "runwayml/stable-diffusion-v1-5" \
    --tick 2 \
    --snap 50 \
    --dump 100 \
    --lr 0.000001 \
    --glr 0.000001 \
    --duration 10 \
    --enable_xformers 1 \
    --gradient_checkpointing 0 \
    --metrics 'fid30k_full' \
    --ema 0 #0.05
    #--nosubdir  \
    #--metrics 'fid_30k_full' \
    #--metrics 'fid_clip_30k_full' \
    #--ema 0.05

elif [ "$model" = 'sd2.1_base_fp16' ]; then
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
    --fp16 1 \
    --batch-gpu 1 \
    --sd_model "stabilityai/stable-diffusion-2-1-base" \
    --tick 2 \
    --snap 50 \
    --dump 100 \
    --lr 0.000001 \
    --glr 0.000001 \
    --duration 10 \
    --enable_xformers 1 \
    --gradient_checkpointing 0 \
    --ema 0 #0.05
    #--nosubdir  \
    #--metrics 'fid_30k_full' \
    #--metrics 'fid_clip_30k_full' \
    #--ema 0.05
else
    echo "Invalid dataset specified"
    exit 1
fi
           
            
        
          

