
how_many=30000
ref_data="coco2014"
ref_dir="/data/datasets/COCO2014"
ref_type="val2014"
eval_res=256
batch_size=128
#batch_size=64
clip_model="ViT-G/14"


#make sure placing `captions.txt' and the correspondong fake_images folder inside fake_dir

#fake_dir='image_experiment/sd1.5_kappa1.5'
fake_dir='image_experiment/sd1.5_kappa1.5_traininglonger'
# fake_dir='image_experiment/sd1.5_kappa2'  
# fake_dir='image_experiment/sd1.5_kappa3'  
# fake_dir='image_experiment/sd1.5_kappa4.5'
# fake_dir='image_experiment/sd2.1base_kappa1.5'
# fake_dir='image_experiment/sd2.1base_kappa2'  
# fake_dir='image_experiment/sd2.1base_kappa3'  
# fake_dir='image_experiment/sd2.1base_kappa4.5'
        

CUDA_VISIBLE_DEVICES=1 python3 evaluation.py --how_many $how_many --ref_data $ref_data --ref_dir $ref_dir --ref_type $ref_type --fake_dir $fake_dir --eval_res $eval_res --batch_size $batch_size --clip_model $clip_model