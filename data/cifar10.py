from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch

from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

general_transform = {
    'train': transform_train,
    'test': transform_valid
}

simclr_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=2),
        transforms.ToTensor(),
        transforms.Normalize([0.4194, 0.4822, 0.4465], [0.2023, 0.1993, 0.2010])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4194, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
}


class CIFAR10Pair(CIFAR10):

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class SplitCifar10(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:

        super(SplitCifar10, self).__init__(root, train=train, transform=transform,
                                           target_transform=target_transform, download=download)

        self.split_idx = None

    def set_split(self, split):
        np_target = np.array(self.targets)
        split_idxs = np.isin(np_target, split)
        np_target = np.where(split_idxs == True)[0]

        self.split_idx = np_target

    def __getitem__(self, index):
        assert self.split_idx is not None
        index = self.split_idx[index]

        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        assert self.split_idx is not None

        return len(self.split_idx)


class SplitSimCIFAR10(CIFAR10):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 ) -> None:
        super(SplitSimCIFAR10, self).__init__(root, train, transform, target_transform,
                                              download)
        self.split_idx = None

    def set_split(self, split):
        np_target = np.array(self.targets)
        split_idxs = np.isin(np_target, split)
        np_target = np.where(split_idxs == True)[0]

        self.split_idx = np_target

    def __getitem__(self, index) -> Tuple[Any, Any, Any]:
        assert self.split_idx is not None
        index = self.split_idx[index]

        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return [pos_1, pos_2], target

    def __len__(self):
        assert self.split_idx is not None
        return len(self.split_idx)
