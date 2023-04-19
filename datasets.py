import torch
from torchvision import datasets
from PIL import Image

class I_CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class I_CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class DatasetPoisoning(object):
    def __init__(self, delta_weight, delta, mode='classwise'):
        self.delta_weight = delta_weight
        self.delta = delta
        self.mode = mode

    def __call__(self, img, target, index):
        if self.mode == 'classwise':
            return torch.clamp(
                img + self.delta_weight * torch.clamp(self.delta[target], min=-1., max=1.), min=0., max=1.
            )
        elif self.mode == 'samplewise':
            return torch.clamp(
                img + self.delta_weight * torch.clamp(self.delta[index], min=-1., max=1.), min=0., max=1.
            )
        else:
            raise ValueError(self.mode)

    def __repr__(self):
        return "Adding pretrained noise to dataset (using poisoned dataset) when re-training"

class P_CIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            for t in self.transform:
                if isinstance(t, DatasetPoisoning):
                    img = t(img, target, index)
                else:
                    img = t(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class P_CIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            for t in self.transform:
                if isinstance(t, DatasetPoisoning):
                    img = t(img, target, index)
                else:
                    img = t(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SeparateTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target, index):
        for t in self.transform:
            if isinstance(t, DatasetPoisoning):
                img = t(img, target, index)
            else:
                img = t(img)
        return img

class P_CIFAR10_TwoCropTransform(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [sep_transform(img, target, index), sep_transform(img, target, index)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

class P_CIFAR100_TwoCropTransform(datasets.CIFAR100):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            sep_transform = SeparateTransform(self.transform)
            img = [sep_transform(img, target, index), sep_transform(img, target, index)]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index