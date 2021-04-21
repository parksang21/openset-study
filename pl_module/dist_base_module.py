from typing import Any
import pytorch_lightning as pl
import torch.nn as nn

from data.cifar10 import CIFAR10, general_transform

from torch.utils.data import DataLoader
from model import resnet18


class LitDistModule(pl.LightningModule):
    def __init__(self, batch_size, lr, num_workers,
                data_root,
                 **kwargs):
        super(LitDistModule, self).__init__()
        self.model = resnet18(with_fc=False)

        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.data_root = data_root


    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--data_root", type=str, default='/datasets')

        return parser

    def train_dataloader(self) -> Any:
        dataset = CIFAR10(self.data_root, train=True,
                          transform=general_transform['train'])
        return DataLoader(dataset, shuffle=True, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return None

