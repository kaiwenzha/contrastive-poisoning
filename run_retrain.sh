#!/bin/bash

set -euxo pipefail

# CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8

# origin
# python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --dataset cifar10 \
#     --arch resnet18 \
#     --cl_alg [SimCLR/MoCov2/BYOL] \
#     [--classwise or --samplewise] \
#     --delta_weight $[8. / 255] \
#     --folder_name eval_poisons \
#     --epochs 1000 \
#     --eval_freq 100 \
#     --pretrained_delta pretrained_poisons/cifar10_res18_simclr_cps.pth \
#     --sas_subset_indices ../sas-data-efficient-contrastive-learning/final_subsets/cifar10-cl-core-idx.pkl

ALGORITHM="MoCov2"
EPOCHS=100
EVAL_FREQ=50
PRETRAINED_DELTA="pretrained_poisons/cifar10_res18_mocov2_cps.pth"
# SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/final_subsets/cifar10-sas-core-selected-0.8-idx.pkl"
SAS_SUBSET_INDICES="../sas-data-efficient-contrastive-learning/final_subsets/cifar10-rand-selected-0.8-idx.pkl"

python -m torch.distributed.launch --nproc_per_node=2 main.py \
    --dataset cifar10 \
    --arch resnet18 \
    --cl_alg "${ALGORITHM}" \
    --samplewise \
    --folder_name eval_poisons \
    --epochs "${EPOCHS}" \
    --eval_freq "${EVAL_FREQ}" \
    --pretrained_delta "${PRETRAINED_DELTA}" \
    --sas_subset_indices "${SAS_SUBSET_INDICES}"

# CIFAR-100
# python -m torch.distributed.launch --nproc_per_node=2 main.py \
#     --dataset cifar100 \
#     --arch resnet18 \
#     --cl_alg BYOL \
#     --samplewise \
#     --folder_name eval_poisons \
#     --epochs 50 \
#     --eval_freq 10 \
#     --pretrained_delta pretrained_poisons/ \
#     --sas_subset_indices ../sas-data-efficient-contrastive-learning/examples/cifar10/cifar10-0.20-sas-subset-indices.pkl
