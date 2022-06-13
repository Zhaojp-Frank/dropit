#python 

CUDA_VISIBLE_DEVICES=0,1 \
  python train.py \
  --cfg configs/imagenet/vit_b_fastminkx0.9.yaml \
  NUM_GPUS 2

