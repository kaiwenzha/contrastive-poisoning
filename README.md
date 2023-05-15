# Contrastive Poisoning

[Project Page](https://contrastive-poisoning.csail.mit.edu/) | [Paper](https://arxiv.org/abs/2202.11202) | [BibTex](assets/bibtex.txt)

This repo contains the official PyTorch implementation of [Indiscriminate Poisoning Attacks on Unsupervised Contrastive Learning](https://arxiv.org/abs/2202.11202) (ICLR 2023 Spotlight, Notable-top-25%), by [Hao He](http://people.csail.mit.edu/hehaodele/)\*, [Kaiwen Zha](https://kaiwenzha.github.io/)\*, [Dina Katabi](http://people.csail.mit.edu/dina/) (*co-primary authors).

<img src='assets/teaser.gif'>

## Setup

- Install dependencies using [conda](https://www.anaconda.com/):
   ```bash
   conda create -n contrastive-poisoning python=3.7
   conda activate contrastive-poisoning
   conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
   pip install tensorboard
   pip install pillow==9.0
   pip install gdown
   
   git clone --recursive https://github.com/kaiwenzha/contrastive-poisoning.git
   cd kornia_pil
   pip install -e .
   ```
   In this work, we implemented PIL-based differentiable data augmentations (to match PIL-based torchvision data augmentations) based on [kornia](https://github.com/kornia/kornia), an OpenCV-based differentiable computer vision library.

- Download datasets (CIFAR-10, CIFAR-100):
   ```bash
   source download_cifar.sh
   ```

- Download all of our pretrained poisons (shown in the table below):
  ```bash
  gdown https://drive.google.com/drive/folders/1FeIHf_tD1bL776Q0PHWGI_rcAkmvQ2iE\?usp\=share_link --folder
  ```

## Pretrained Poisons

### CIFAR-10

<table class="tg">
<thead>
  <tr>
    <th rowspan="2">Attacker Type</th>
    <th colspan="3">Victim's Algorithm</th>
  </tr>
  <tr>
    <th>SimCLR</th>
    <th>MoCo v2</th>
    <th>BYOL</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">CP-S</td>
    <td align="center">44.9 / <a href="https://drive.google.com/file/d/1Wfs6VVh1-ZqzI3Bnt1OG4m5eU9mtk0am/view?usp=share_link">poison</a></td>
    <td align="center">55.1 / <a href="https://drive.google.com/file/d/1sUbhqqlDAqmpVcnBdgzdDvd1pFrflZ12/view?usp=share_link">poison</a></td>
    <td align="center">59.6 / <a href="https://drive.google.com/file/d/1lMCro51QEfzs6L7Uel17L701VznRKJVR/view?usp=share_link">poison</a></td>
  </tr>
  <tr>
    <td align="center">CP-C</td>
    <td align="center">68.0 / <a href="https://drive.google.com/file/d/1rFLlUvRpSTgMBTFNi3yPxb-Jc5xZpZeD/view?usp=share_link">poison</a></td>
    <td align="center">61.9 / <a href="https://drive.google.com/file/d/1dR5BAhwU3KgfWmsonVupJLurF8DDUAHS/view?usp=share_link">poison</a></td>
    <td align="center">56.9 / <a href="https://drive.google.com/file/d/1fXaBHNBsG8IPyU9tO6ErBdSgqGrwoyAG/view?usp=share_link">poison</a></td>
  </tr>
</tbody>
</table>

*The results in the table above assume the victim's algorithm being known to the attacker, i.e., the attacker and the victim are using the same CL algorithm.*

*BYOL performance may slightly differ from what is reported in the table/paper above because we have replaced the implementation of synchronized batch normalization from the previous `apex.parallel.SyncBatchNorm` (now deprecated) to `torch.nn.SyncBatchNorm` when releasing the code.*

To evaluate our pretrained poisons, re-train the corresponding CL model on the poisoned dataset by running
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 main.py \
      --dataset cifar10 \
      --arch resnet18 \
      --cl_alg [SimCLR/MoCov2/BYOL] \
     [--classwise or --samplewise] \
      --delta_weight $[8./255] \
      --folder_name eval_poisons \
      --epochs 1000 \
      --eval_freq 100 \
      --pretrained_delta pretrained_poisons/xxx.pth
```
Set arguments `--cl_alg`, `--classwise` or `--samplewise`, and `--pretrained_delta` according to the evaluated poison you choose before running. Taking the SimCLR CP-S poison ([`cifar10_res18_simclr_cps.pth`](https://drive.google.com/file/d/1Wfs6VVh1-ZqzI3Bnt1OG4m5eU9mtk0am/view?usp=share_link)) as an example, the running script should set `--cl_alg SimCLR`, `--samplewise`, and `--pretrained_delta pretrained_poisons/cifar10_res18_simclr_cps.pth`.

## Training
This code supports training on CIFAR-10 and CIFAR-100. 

### Contrastive Learning Baselines
To train a contrastive learning (CL) model (e.g., SimCLR, MoCov2, BYOL) on the clean dataset, run
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 main.py \
      --dataset cifar10 \
      --arch resnet18 \
      --cl_alg [SimCLR/MoCov2/BYOL] \
      --folder_name baseline \
      --baseline \
      --epochs 1000 \
      --eval_freq 100
```

### Class-wise Contrastive Poisoning (CP-C)
1. Run CP-C to generate the class-wise poison
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 main.py \
          --dataset cifar10 \
          --arch resnet18 \
          --cl_alg [SimCLR/MoCov2/BYOL] \
          --classwise \
          --delta_weight $[8./255] \
          --folder_name CP_C \
          --epochs 1000 \
          --eval_freq 10000 \
          --print_freq 5 \
          --num_steps 1 \
          --step_size 0.1 \
          --model_step 20 \
          --noise_step 20 \
         [--allow_mmt_grad]
    ```
    Add `--allow_mmt_grad` flag to enable dual-branch propagation when running on MoCov2 and BYOL.

2. Re-train the CL model (e.g., SimCLR, MoCov2, BYOL) on the poisoned dataset generated by CP-C
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 main.py \
          --dataset cifar10 \
          --arch resnet18 \
          --cl_alg [SimCLR/MoCov2/BYOL] \
          --classwise \
          --delta_weight $[8./255] \
          --folder_name CP_C \
          --epochs 1000 \
          --eval_freq 100 \
          --pretrained_delta <.../last.pth>
    ```
    `--pretrained_delta` is the path to the model checkpoint from step 1, which contains the generated poison.

### Sample-wise Contrastive Poisoning (CP-S)
1. Run CP-S to generate the sample-wise poison
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 main.py \
          --dataset cifar10 \
          --arch resnet18 \
          --cl_alg [SimCLR/MoCov2/BYOL] \
          --samplewise \
          --delta_weight $[8./255] \
          --folder_name CP_S \
          --epochs 200 \
          --eval_freq 10000 \
          --num_steps 5 \
          --step_size 0.1 \
          --initialized_delta <.../last.pth or pretrained_poisons/cifar10_res18_xxx_cpc.pth> \
         [--allow_mmt_grad]
    ```
    - To get a stronger poison, here we use learned class-wise poison to initialize the sample-wise poison. `--initialized_delta` can either be set as the path to the model checkpoint trained by CP-C step 1, or use our generated CP-C poison in `pretrained_poisons` folder (Note: the CL algorithm should be matched).
    - Add `--allow_mmt_grad` flag to enable dual-branch propagation when running on MoCov2 and BYOL.

2. Re-train the CL model (e.g., SimCLR, MoCov2, BYOL) on the poisoned dataset generated by CP-S
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m torch.distributed.launch --nproc_per_node=4 main.py \
          --dataset cifar10 \
          --arch resnet18 \
          --cl_alg [SimCLR/MoCov2/BYOL] \
          --samplewise \
          --delta_weight $[8./255] \
          --folder_name CP_S \
          --epochs 1000 \
          --eval_freq 100 \
          --pretrained_delta <.../last.pth> (for MoCov2 and BYOL) or <.../ckpt_epoch_160.pth> (for SimCLR)
    ```
    `--pretrained_delta` is the path to the model checkpoint from step 1, which contains the generated poison.

### Model Resuming
To resume any interrupted model trained above, keep all commands unchanged and simply add `--resume <.../curr_last.pth>`, which should specify the full path to the latest checkpoint (`curr_last.pth`) of the interrupted model. 

## Acknowledgements
This code is partly based on the open-source implementations from [SupContrast](https://github.com/HobbitLong/SupContrast), [MoCo](https://github.com/facebookresearch/moco), [lightly](https://github.com/lightly-ai/lightly) and [kornia](https://github.com/kornia/kornia).

## Citation
If you use this code for your research, please cite our paper:
```bibtex
@inproceedings{he2023indiscriminate,
    title={Indiscriminate Poisoning Attacks on Unsupervised Contrastive Learning},
    author={Hao He and Kaiwen Zha and Dina Katabi},
    booktitle={The Eleventh International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=f0a_dWEYg-Td}
}
```
