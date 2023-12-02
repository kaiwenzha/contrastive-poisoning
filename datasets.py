import torch
from torchvision import datasets
from PIL import Image
import pickle
from typing import Callable, Optional, List


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
    def __init__(self, delta_weight, delta, mode="classwise"):
        self.delta_weight = delta_weight
        self.delta = delta
        self.mode = mode

    def __call__(self, img, target, index):
        if self.mode == "classwise":
            return torch.clamp(
                img
                + self.delta_weight
                * torch.clamp(self.delta[target], min=-1.0, max=1.0),
                min=0.0,
                max=1.0,
            )
        elif self.mode == "samplewise":
            return torch.clamp(
                img
                + self.delta_weight * torch.clamp(self.delta[index], min=-1.0, max=1.0),
                min=0.0,
                max=1.0,
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


class P_CIFAR10_TwoCropTransform_SAS_Subset(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        subset_indices: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        super(P_CIFAR10_TwoCropTransform_SAS_Subset, self).__init__(
            root=root, train=train, transform=transform, download=download
        )
        with open(subset_indices, "rb") as f:
            raw = pickle.load(f)
            if isinstance(raw, dict):
                self.subset_indices: List[int] = raw["indices"]
            else:
                self.subset_indices: List[int] = raw
        print(f'Using SAS indices from {subset_indices}')

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, subset_index):
        index = self.subset_indices[subset_index]
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
