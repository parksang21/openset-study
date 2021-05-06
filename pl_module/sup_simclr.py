from typing import Any
import pytorch_lightning as pl
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cifar10 import SplitSimCIFAR10, simclr_transform, CIFAR10, general_transform
from model.custom_resnet import model_dict
from model.resnet_big import SupConResNet
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

        # m, self.emb_dim = model_dict[model]
        # self.model = m(with_fc=False)
        self.model = SupConResNet()
        # self.head = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.emb_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.emb_dim, feat_dim)
        # )
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

    def test_dataloader(self):
        data = SplitSimCIFAR10(root=self.data_root,
                               train=False,
                               transform=simclr_transform['train'])

        data.set_split([i for i in range(10)])

        loader = DataLoader(data, shuffle=False,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
        self.len_test_loader = len(loader)
        return loader

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay,
                               momentum=self.momentum)

    def training_step(self, batch, batch_idx):
        x1, y = batch
        bt_size = y.size(0)

        self.warmup_learning_rate(self.current_epoch, batch_idx,
                                  self.len_train_loader, self.optimizers())

        feat = self.forward(torch.cat(x1, dim=0))

        f1, f2 = torch.split(feat, [bt_size, bt_size], dim=0)
        feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = self.criterion(feat, y)

        self.log("losses", loss)

        return loss

    def test_step(self, batch, batch_idx):
        '''
        in testing section, it only takes single input (1_views)
        :param batch:
        :param batch_idx:
        :return:
        '''
        x, y = batch

        proj_feat = self.forward(torch.cat(x, dim=0))

        f1, f2 = torch.split(proj_feat, [y.size(0), y.size(0)], dim=0)
        proj_feat = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        known_idx = torch.stack([i == y for i in self.split_class]).sum(0).bool()

        known_loss = self.criterion(proj_feat[known_idx], y[known_idx])
        unknown_loss = self.criterion(proj_feat[~known_idx], y[~known_idx])

        feat = self.model.encoder(torch.cat(x, dim=0))

        # save the results of testing
        self.target.append(y.detach().cpu().numpy())
        self.projection.append(proj_feat.detach().cpu().numpy())
        self.feature.append(feat.detach().cpu().numpy())

        self.log("known_loss", known_loss)
        self.log("unknown_loss", unknown_loss)

        return known_loss

    def on_test_start(self):
        self.feature = list()
        self.projection = list()
        self.sim = list()
        self.target = list()


    def on_test_end(self):
        import pickle as pkl

        print(f"{'='*20}\n"
              f"[[length]]\n"
              f"target : {len(self.target)}\n"
              f"projection : {len(self.projection)}\n"
              f"feat : {len(self.feature)}\n")

        target = np.concatenate(self.target, axis=0)
        feature = np.concatenate(self.feature, axis=0)
        projection = np.concatenate(self.projection, axis=0)
        classes = self.split_class

        save_dict = {
            'target': target,
            'feature': feature,
            'projection': projection,
            'class': classes
        }

        with open(self.logger.log_dir + "/data.pkl", "wb") as f:
            pkl.dump(save_dict, f)


        # known_mask = np.isin(target, self.split_class)
        # known_idx = target[known_mask]
        # unknown_idx = target[~known_mask]
        #
        # known_proj = projection[known_idx]
        # print(known_proj.shape)
        # # cat_proj = np.concatenate(known_proj[:,0,:], known_proj[:, 1, :], axis=0)

        # assert sum(known_mask) == len(known_idx)


    def on_train_epoch_start(self):
        self.adjust_learning_rate()

    def forward(self, x):
        # out, _ = self.model(x)
        # out = F.normalize(self.head(out), dim=1)
        out = self.model(x)
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
