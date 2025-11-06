
![SiD-LSG](SiD-LSG-teaser.png "One-Step Text-to-Image Generation")

# Text-to-Image Diffusion Distillation with SiD-LSG

This **SiD-LSG** repository contains the code and model checkpoints necessary to replicate the findings of our ICLR 2025 paper: [Guided Score identity Distillation for Data-Free One-Step Text-to-Image Generation](https://arxiv.org/abs/2406.01561). Note that this paper was originally titled **"Long and Short Guidance in Score Identity Distillation for One-Step Text-to-Image Generation"** and was first posted on arXiv in June 2024, alongside the release of the corresponding code and model checkpoints. The technique, Long and Short Guidance (**LSG**), is used with Score identity Distillation (**SiD**: [ICML 2024 paper](https://arxiv.org/abs/2404.04057), [Code](https://github.com/mingyuanzhou/SiD)) to distill Stable Diffusion models for one-step text-to-image generation in a data-free manner.

We are actively developing an improved version of **SiD-LSG**, which will be placed in a separate branch and introduce the following enhancements (Update: The features described below are now available in [sid-dit](https://github.com/apple/ml-sid-dit). Our team at UT Austin is also preparing to release the SiD-DiT distilled checkpoints for SANA, SD3-Medium, SD3.5-Medium, SD3.5-Large, and FLUX.1-dev on Hugging Face.):  

1. **AMP Support** – Leverages automatic mixed precision (AMP) to significantly reduce memory usage and improve speed compared to the current FP32 default, with minimal impact on performance.
2. **FSDP + AMP Integration** – Supports much larger models by combining Fully Sharded Data Parallel (FSDP) with AMP. Our implementation relies solely on native PyTorch libraries, avoiding third-party containers to ensure maximum flexibility for code customization.  
3. **Diffusion GAN Integration** – Building on the success of SiDA ([ICLR 2025 paper](https://arxiv.org/abs/2410.14919), [Code](https://github.com/mingyuanzhou/SiD/tree/sida)), which achieves state-of-the-art performance in distilling EDM and EDM2 models using a single generation step and without requiring CFG, we will integrate adversarial training from diffusion GANs ([ICLR 2023 paper](https://arxiv.org/abs/2206.02262), [Code](https://github.com/Zhendong-Wang/Diffusion-GAN)) into guided SiD. This enhancement will significantly improve the trade-off between reducing FID (better diversity) and increasing CLIP scores (better text-image alignment), all without introducing any additional model parameters.
4. **A New Guidance Strategy** – Introduces a novel guidance method with lower memory and computational requirements than LSG, while achieving comparable performance without the need for tuning the guidance scale.
5. **Multistep Distillation** – Enhances performance by enabling the distillation of multi-step generators. Note this was already implemented in the current code, but some adjustments are needed to unlock its potential.

If you find our work useful or incorporate our findings in your own research, please consider citing our paper:

 - **SiD-LSG**:
```bibtex
@inproceedings{zhou2025guided,
title={Guided Score identity Distillation for Data-Free One-Step Text-to-Image Generation},
author={Mingyuan Zhou and Zhendong Wang and Huangjie Zheng and Hai Huang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://arxiv.org/abs/2406.01561}
}
```

Our work on SiD-LSG builds on prior research on SiD. If relevant, you may also consider citing the following:
- **SiD**:
```bibtex
@inproceedings{zhou2024score,
  title={Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation},
  author={Mingyuan Zhou and Huangjie Zheng and Zhendong Wang and Mingzhang Yin and Hai Huang},
  booktitle={International Conference on Machine Learning},
  url={https://arxiv.org/abs/2404.04057},
  year={2024}
}
```


## State-of-the-art Performance
SiD-LSG functions as a data-free distillation method capable of generating photo-realistic images in a single step. By employing a relatively low guidance scale, such as 1.5, it surpasses the teacher stable diffusion model in achieving lower zero-shot Fréchet Inception Distances (FID). This comparison involves 30k COCO2014 caption-prompted images against the COCO2014 validation set, though it does so at the cost of a reduced CLIP score.

The one-step generators distilled with SiD-LSG achieve the following FID and CLIP scores:

<table>
  <tr>
    <th rowspan="6">Stable Diffusion 1.5</th>
    <th>Guidance Scale</th>
    <th>FID</th>
    <th>CLIP</th>
  </tr>
    <tr><td>1.5</td><td>8.71</td><td>0.302</td></tr>
  <tr><td>1.5 (longer training)</td><th>8.15</th><td>0.304</td></tr>
  <tr><td>2</td><td>9.56</td><td>0.313</td></tr>
  <tr><td>3</td><td>13.21</td><td>0.314</td></tr>
  <tr><td>4.5</td><td>16.59</td><td>0.317</td></tr>
  <tr>
    <th rowspan="5">Stable Diffusion 2.1-base</th>
    <th>Guidance Scale</th>
    <th>FID</th>
    <th>CLIP</th>
  </tr>
  <tr><td>1.5</td><td>9.52</td><td>0.308</td></tr>
  <tr><td>2</td><td>10.97</td><td>0.318</td></tr>
  <tr><td>3</td><td>13.50</td><td>0.321</td></tr>
  <tr><td>4.5</td><td>16.54</td><th>0.322</th></tr>
</table>

          
                
## Installation

To install the necessary packages and set up the environment, follow these steps:

### Prepare the Code and Conda Environment

First, clone the repository to your local machine:

```bash
git clone https://github.com/mingyuanzhou/SiD-SLG.git
cd SiD-LSG
```
To create the Conda environment with all the required dependencies and activate it, run:

```bash
conda env create -f sid_lsg_environment.yml
conda activate sid_lsg
```




### Prepare the Datasets

To train the model, you need to provide training prompts. By default, we use **Aesthetic6+**, but you can also choose **Aesthetic6.25+**, **Aesthetic6.5+**, or any other list of prompts, as long as they do not include COCO captions. 

To obtain the **Aesthetic6+** prompts from Hugging Face, follow their guidelines. Once you have the prompts, save them in the following path:  
`/data/datasets/aesthetics_6_plus/aesthetics_6_plus.txt`.

Alternatively, you can download the prompts directly from [this link](https://huggingface.co/UT-Austin-PML/SiD-LSG/) and extract the `.tar` file to the specified directory.

To evaluate the zero-shot FID of the distilled one-step generator, you will first need to download the COCO2014 validation set from [COCOdataset](https://cocodataset.org/#download), and then prepare the COCO2014 validation set using the following command:
```
python cocodataset_tool.py --source=/path/to/COCO \
 --dest=MS-COCO-256 --resolution=256x256 --transform='center-crop' --phase='val'
```

Once prepared, place them into the `/data/datasets/MS-COCO-256/val` folder.

To make an apple-to-apple comparison with previous methods such as [GigaGAN](https://mingukkang.github.io/GigaGAN/), you may use the `captions.txt`, obtained from [GigaGAN/COCOevaluation](https://github.com/mingukkang/GigaGAN/blob/main/evaluation/scripts/evaluate_GigaGAN_t2i_coco256.sh), to generate 30k images and use them to compute the zero-shot COCO2014 FID. 


## Usage



### Training
After activating the environment, you can run the scripts or use the modules provided in the repository. Example:

```bash
sh run_sid.sh 'sid1.5'
```

Adjust the --batch-gpu parameter according to your GPU memory limitations. To save memory, such as fitting GPU with 24GB memomry, you may set `--ema 0` to turn off EMA and set `--fp16 1` to use mixed-precision training. 

### Checkpoints of SiD-LSG one-step generators 

The one-step generators produced by SiD-LSG are provided in [huggingface/UT-Austin-PML/SiD-LSG](https://huggingface.co/UT-Austin-PML/SiD-LSG/tree/main)

You can first download the SiD-LSG one-step generators and place them into `/data/Austin-PML/SiD-LSG/` or a folder you choose. Alternatively, you can replace `/data/Austin-PML/SiD-LSG/` with 'https://huggingface.co/UT-Austin-PML/SiD-LSG/resolve/main/'  to directly download the checkpoint from HuggingFace

### Generate example images


Generate examples images using user-provided prompts and random seeds:

 - Reproduce Figure 1:

```bash
python generate_onestep.py --outdir='image_experiment/example_images/figure1' --seeds='8,8,2,3,2,1,2,4,3,4' --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'  --text_prompts='prompts/fig1-captions.txt'  --custom_seed=1
```

 - Reproduce Figure 6 (the columns labeled SD1.5 and SD2.1), ensuring the seeds align with the positions of the prompts within the HPSV2 defined list of prompts:
```bash
python generate_onestep.py --outdir='image_experiment/example_images/figure6/sd1.5' --seeds='668,329,291,288,057,165' --batch=6 --network='/data/Austin-PML/SiD-LSG/batch512_cfg4.54.54.5_t625_8380_v2.pkl' --text_prompts='prompts/fig6-captions.txt' --custom_seed=1
```

```bash
python generate_onestep.py --outdir='image_experiment/example_images/figure6/sd2.1base' --seeds='668,329,291,288,057,165' --batch=6 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'  --text_prompts='prompts/fig6-captions.txt' --custom_seed=1
```

 - Reproduce Figure 8:
```bash    
python generate_onestep.py --outdir='image_experiment/example_images/figure8' --seeds='4,4,1,1,4,4,1,1,2,7,7,6,1,20,41,48' --batch=16 --network='/data/Austin-PML/SiD-LSG/batch512_sd21_cfg4.54.54.5_t625_7168_v2.pkl' --repo_id='stabilityai/stable-diffusion-2-1-base'  --text_prompts='prompts/fig8-captions.txt' --custom_seed=1
```

### Evaluations


- Generation: Generate 30K images to calculate zeroshot COCO FID (see the comments inside  [generate_onestep.py]([https://github.com/mingyuanzhou/SiD-LSG/generate_onestep.py](https://github.com/mingyuanzhou/SiD-LSG/blob/main/generate_onestep.py)) for more detail):

```bash
#SLG guidance scale kappa1=kappa2=kappa3=kappa4 = 1.5, longer training
#FID 8.15, CLIP 0.304     
torchrun --standalone --nproc_per_node=4 generate_onestep.py --outdir='image_experiment/sid_sd15_runs/sd1.5_kappa1.5_traininglonger/fake_images' --seeds=0-29999 --batch=16 --network='https://huggingface.co/UT-Austin-PML/SiD-LSG/resolve/main/batch512_cfg1.51.51.5_t625_18789_v2.pkl'  
```




- Computing evaluation metrics: Following GigaGAN to compute FID and CLIP using the 30k images generated with [generate_onestep.py](https://github.com/mingyuanzhou/SiD-LSG/blob/main/generate_onestep.py); you also need to place `captions.txt` into  the user defined path for `fake_dir`

Download [GigaGAN/evaluation](https://github.com/mingukkang/GigaGAN/tree/main/evaluation) 

Place `evaluate_SiD_t2i_coco256.sh` into its folder: `GigaGAN/evaluation/scripts`

Modify `fake_dir=` inside `evaluate_SiD_t2i_coco256.sh` to point to the folder that consits of `captions.txt` and the `fake_images` folder with 30k fake images, and run:

```bash
bash scripts/evaluate_SiD_t2i_coco256.sh
```



### Acknowledgements

The SiD-LSG code integrates functionalities from [Hugging Face/Diffusers](https://huggingface.co/docs/diffusers/en/index) into the [mingyuanzhou/SiD](https://github.com/mingyuanzhou/SiD) repository, which was build on [NVlabs/edm](https://github.com/NVlabs/edm) and [pkulwj1994/diff_instruct](https://github.com/pkulwj1994/diff_instruct). 





## Contributing to the Project

### Code Contributions
- **Mingyuan Zhou**: Led the project, debugged and developed the integration of Stable Diffusion and Long-Short Guidance into the SiD codebase, wrote the evaluation pipelines, and performed the exerpiments. 
- **Zhendong Wang** Led the effort of integrating Stable Diffusion into the SiD codebase.
- **Huangjie Zheng** Led the effort of evaluating the generation results and preparing the COCO dataset.
- **Hai Huang**: Led the effort in adapting the code for Google's internal computing infrasturcture.
- **Michael (Qijia) Zhou**, Led the effort in preparing the data and participated in adapting the code to Google's internal computing infrasturcture.
- All contributors worked closely together to co-develop essential components and writing various subfunctions.


To contribute to this project, follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Contact

If you want to contact me, you can reach me at `mingyuanzhou@gmail.com`.

## License

This project uses the following license: [Apache-2.0 license](README.md).

