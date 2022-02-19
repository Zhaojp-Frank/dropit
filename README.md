# DropIT: <u>Drop</u>ping <u>I</u>ntermediate <u>T</u>ensors for Memory-Efficient DNN Training

DropIT aims to reduce memory consumption in intermediate tensors caching during training DNNs. In computer vision tasks, the GPU memory of intermediate tensors is often hundreds of times the model size (e.g., 20 GB vs. 100 MB for ResNet-50). DropIT solves this problem by adaptively caching part of intermediate tensors in the forward pass, and recovering sparsified tensors for gradient computation in the backward pass.

![image](https://user-images.githubusercontent.com/20626415/154811392-61b45fba-fe1c-4df1-a9ec-865cf12bd8ad.png)

Interested? Take a look at our paper: 

[DropIT: <u>Drop</u>ping <u>I</u>ntermediate <u>T</u>ensors for Memory-Efficient DNN Training](http) 

Joya Chen*, Kai Xu*, Yifei Cheng, Angela Yao (* Equal Contribution)

This repository contains the implementation of DropIT, which is co-developed by [Joya Chen](https://github.com/ChenJoya/) and [Kai Xu](https://github.com/kai422).

## Install

The installation of DropIT is simple. The implementation only relies on [PyTorch](https://pytorch.org/) and [PyTorch-Lightning](https://www.pytorchlightning.ai/).

```shell
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch # other cuda version is also okay

pip install pytorch-lightning lightning-bolts 

git clone https://github.com/ChenJoya/dropit

pip install -e . 
```

## Train

We provide configs in [dropit/configs](https://github.com/ChenJoya/dropit/tree/main/configs). For example, training vision transformer ViT-B/16 in ImageNet:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --cfg configs/imagenet/vit_b_fastminkx0.9.yaml NUM_GPUS 2
```

## Evaluation

Evaluation will be performed every N (N can be set in config) epochs during training. You can use tensorboard to see the results. An example:

![tensorboard](https://user-images.githubusercontent.com/20626415/154811411-3481dc61-9870-4c0a-a67e-20e217abf634.png)

## Performance

### CIFAR-100

|  Model   | Top-1 Acc | Top-5 Acc | Cache (MB) |
|  ----  | ----  |  ----  | ---- |
| ResNet-18 (32x32) | 77.96 | 94.05 | 648 |
| ResNet-18 (32x32) w. DropIT | **78.17** | **94.19** | **598** |
| ViT-B/16 (224x224) | 90.32 | 98.88 | 20290 |
| ViT-B/16 (224x224) w. DropIT | **90.90** | **99.02** | **16052** |

### ImageNet-1k

|  Model   | Top-1 Acc | Top-5 Acc | Cache (MB) |
|  ----  | ----  |  ----  | ---- |
| ResNet-18 (224x224) | 69.76 | 89.08 | 2826 |
| ResNet-18 (224x224) w. DropIT | **69.85** | **89.39** | **2600** |
| ViT-B/16 (224x224) | 83.40 | 96.96 | 20290 |
| ViT-B/16 (224x224) w. DropIT | **83.61** | **97.01** | **16056** |

## Citation

Please consider citing our paper if it helps your research. The following is a BibTeX reference. 

```
{}
```
