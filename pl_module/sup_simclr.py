from typing import Any
import pytorch_lightning as pl
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cifar10 import SplitSimCIFAR10, simclr_transform
from model.custom_resnet import model_dict
from losses.contrastive import SupConLoss


class SupSimModule(pl.LightningModule):
    def __init__(self, lr, batch_size, num_workers, model,
                 temperature,
                 split_class,
                 data_root,
                 weight_decay,
                 feat_dim,
                 momentum,
                 max_epochs,
                 **kwargs):
        super(SupSimModule, self).__init__()

        m, self.emb_dim = model_dict[model]
        self.model = m(with_fc=False)
        self.head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dim, feat_dim)
        )
        self.feat_dim = feat_dim

        self.criterion = SupConLoss(temperature=temperature)

        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.split_class = split_class
        self.data_root = data_root
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_decay_rate = 0.1
        self.max_epochs = max_epochs

        self.warmup_from = 0.01
        self.warm_epochs = 10
        self.eta_min = self.lr * (self.lr_decay_rate ** 3)
        self.warmup_to = self.eta_min + (self.lr - self.eta_min) * (
            1 + math.cos(math.pi * self.warm_epochs / self.max_epochs)) / 2

        self.train_accuracy = pl.metrics.Accuracy()

        self.save_hyperparameters()

    @staticmethod
    def add_module_specific_args(parser):
        parser.add_argument("--lr", type=float, default=5e-1)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--temperature", type=float, default=0.5)
        parser.add_argument("--split_class", nargs='+', default=[0, 1, 2, 3, 4, 5])
        parser.add_argument("--data_root", type=str, default='/datasets')
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--model", type=str, default='resnet50')
        parser.add_argument("--feat_dim", type=int, default=128)
        parser.add_argument("--momentum", type=float, default=0.9)
        return parser

    def train_dataloader(self) -> Any:
        data = SplitSimCIFAR10(root=self.data_root,
                               train=True,
                               transform=simclr_transform['train'],
                               download=True)

        data.set_split(self.split_class)

        loader = DataLoader(data, shuffle=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
        self.len_train_loader = len(loader)
        return loader

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay,
                               momentum=self.momentum)

# todo learning rate warm up has to be added
    def training_step(self, batch, batch_idx):
        x1, y = batch

        bt_size = y.size(0)

        self.warmup_learning_rate(self.current_epoch, batch_idx,
                                  self.len_train_loader, self.optimizers())

        feat = self.forward(torch.cat(x1, dim=0))

        f1, f2 = torch.split(feat, [bt_size, bt_size], dim=0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = self.criterion(feat, y)

        return loss

    def on_epoch_start(self):
        self.adjust_learning_rate()

    def forward(self, x):
        out, _ = self.model(x)
        out = F.normalize(self.head(out), dim=1)
        return out

    def adjust_learning_rate(self):
        lr = self.lr
        eta_min = lr * (self.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * self.current_epoch / self.max_epochs)) / 2
        self.log('lr', lr, prog_bar=True)
        for param_group in self.optimizers().param_groups:
            param_group['lr'] = lr

    def warmup_learning_rate(self, epoch, batch_id, total_batches, optimizer):
        if epoch <= self.warm_epochs:
            p = (batch_id + (epoch - 1) * total_batches) / \
                (self.warm_epochs * total_batches)
            lr = self.warmup_from + p * (self.warmup_to - self.warmup_from)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            self.log('lr', lr, prog_bar=True)