import torch
import torch.nn as nn
import kornia.augmentation as K

from cl_algs import MoCo, BYOL
from resnet import ResNetWithHead, ResNetNoHead, model_dict

from util import log
print_yellow = lambda text: log(text, color='yellow')

class AttackModel(nn.Module):
    def __init__(self, arch, dataset, opt):
        super(AttackModel, self).__init__()
        self.arch = arch
        self.dataset = dataset
        self.opt = opt
        if opt.cl_alg == 'SimCLR':
            self.backbone = ResNetWithHead(arch=arch)
        elif opt.cl_alg.startswith('MoCo'):
            self.backbone = MoCo(
                ResNetWithHead, arch=arch, dim=opt.moco_dim, K=opt.moco_k, m=opt.moco_m, T=opt.temp, mlp=True, allow_mmt_grad=opt.allow_mmt_grad
            )
        elif opt.cl_alg == 'BYOL':
            self.backbone = BYOL(
                ResNetNoHead(arch=arch), num_ftrs=model_dict[arch][1], allow_mmt_grad=opt.allow_mmt_grad
            )
        else:
            raise ValueError(opt.cl_alg)

        if dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif dataset == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        else:
            raise ValueError(dataset)

        normalize = K.Normalize(mean=mean, std=std)

        # differentiable augmentations (can directly apply on batch)
        self.transform = nn.Sequential(
            K.RandomResizedCrop(size=(opt.size, opt.size), scale=(0.2, 1.)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            normalize
        )

        self.initialize_delta()

    def initialize_delta(self, input_delta=None):
        if self.opt.baseline:
            delta_size = 1
        elif self.opt.classwise:
            delta_size = self.opt.n_cls
        else:
            delta_size = self.opt.dataset_size

        if input_delta is None:
            delta = torch.rand(delta_size, 3, self.opt.size, self.opt.size).mul(2.).sub(1.)
            if self.opt.local_rank == 0 and not self.opt.baseline:
                print_yellow(f"Initialize delta shape: {delta.shape}")
        else:
            delta = input_delta
            if self.opt.local_rank == 0:
                print_yellow(f"Initialize delta from input delta, shape: {delta.shape}")

        # make noise learnable
        self.delta = nn.Parameter(delta, requires_grad=True)

    def forward(self, img, index, labels=None):
        bsz = img.shape[0]
        if self.opt.baseline or len(self.opt.pretrained_delta):
            # noise has already been added in dataloader for re-training CL model
            mixed_img = img
            bsz = img.shape[0] // 2
        elif self.opt.classwise:
            # add class-wise noise to image
            mixed_img = torch.clamp(
                img + self.opt.delta_weight * torch.clamp(self.delta[labels], min=-1., max=1.), min=0., max=1.
            )
        elif self.opt.samplewise:
            # add sample-wise noise to image
            mixed_img = torch.clamp(
                img + self.opt.delta_weight * torch.clamp(self.delta[index], min=-1., max=1.), min=0., max=1.
            )
        else:
            raise ValueError('Running scheme is not specified!')

        # data augmentation
        if self.opt.baseline or len(self.opt.pretrained_delta):
            aug1, aug2 = torch.split(mixed_img, [bsz, bsz], dim=0)
        else:
            aug1, aug2 = self.transform(mixed_img), self.transform(mixed_img)
        aug = torch.cat([aug1, aug2], dim=0)

        out_dict = {}
        if self.opt.cl_alg == 'SimCLR':
            features = self.backbone(aug)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            out_dict['features'] = features
        elif self.opt.cl_alg.startswith('MoCo'):
            moco_logits = self.backbone(
                im_q=aug1, im_k=aug2.detach() if not self.opt.allow_mmt_grad else aug2
            )
            out_dict['moco_logits'] = moco_logits
        elif self.opt.cl_alg == 'BYOL':
            y0, y1 = self.backbone(aug1, aug2)
            out_dict['output'] = (y0, y1)
        else:
            raise ValueError(self.opt.cl_alg)

        return out_dict