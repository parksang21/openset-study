from typing import Any
import pytorch_lightning as pl
import math
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cifar10 import general_transform
from data.cifar10 import SplitCifar10, CIFAR10
from model.custom_resnet import model_dict
from losses.contrastive import SupConLoss
from model.resnet_big import SupConResNet

class CESupSim(pl.LightningModule):
    def __init__(self, lr, batch_size, num_workers, model,
                 temperature,
                 split_class,
                 data_root,
                 weight_decay,
                 feat_dim,
                 momentum,
                 max_epochs,
                 path,
                 **kwargs):
        super(CESupSim, self).__init__()

        # m, self.emb_dim = model_dict[model]
        # self.model = m(with_fc=False)
        # self.head = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.emb_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.emb_dim, feat_dim)
        # )
        self.model = SupConResNet()
        state_dict = torch.load(path)
        if kwargs['train']:
            self.load_state_dict(state_dict['state_dict'])
        self.feat_dim = feat_dim

        self.classifier = nn.Linear(2048, 6)

        self.criterion = nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.split_class = split_class
        self.data_root = data_root
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_decay_rate = 0.1
        self.max_epochs = max_epochs
        self.path = path

        self.warmup_from = 0.01
        self.warm_epochs = 10
        self.eta_min = self.lr * (self.lr_decay_rate ** 3)
        self.warmup_to = self.eta_min + (self.lr - self.eta_min) * (
            1 + math.cos(math.pi * self.warm_epochs / self.max_epochs)) / 2

        self.train_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

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

    def on_train_start(self):
        print(self.path)

    def train_dataloader(self) -> Any:
        data = SplitCifar10(root=self.data_root,
                               train=True,
                               transform=general_transform['train'],
                               download=True)

        data.set_split(self.split_class)

        loader = DataLoader(data, shuffle=True,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)

        self.len_train_loader = len(loader)
        return loader

    def configure_optimizers(self):
        return torch.optim.SGD(self.classifier.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay,
                               momentum=self.momentum)

    def forward(self, x):
        out = self.model.encoder(x)
        out = self.classifier(out)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch

        logit = self.forward(x)
        _, y_pred = logit.max(1)

        loss = self.criterion(logit, y)

        train_acc_batch = self.train_accuracy(y_pred, y)
        self.log('train_acc_batch', train_acc_batch, prog_bar=True)

        return loss

    def test_dataloader(self):
        data = SplitCifar10(root=self.data_root,
                       train=False,
                       transform=general_transform['test'])

        data.set_split([i for i in range(10)])

        loader = DataLoader(data,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
        self.len_train_loader = len(loader)

        return loader

    def on_test_start(self):
        self.feature = list()
        self.loss = list()
        self.target = list()
        self.logit = list()

    def test_step(self, batch, batch_idx):
        x, y = batch
        feature = self.model.encoder(x)
        logit = self.classifier(feature)
        _, y_pred = logit.max(1)

        # print(logit)
        #
        # loss = self.criterion(logit, y)

        self.feature.append(feature.detach().cpu().numpy())
        self.logit.append(logit.detach().cpu().numpy())
        self.target.append(y.detach().cpu().numpy())

        test_acc_batch = self.test_accuracy(y_pred, y)

        self.log("test_acc", test_acc_batch)

        return test_acc_batch

    def test_epoch_end(self, outputs):
        feature = np.concatenate(self.feature, axis=0)
        logit = np.concatenate(self.logit, axis=0)
        target = np.concatenate(self.target, axis=0)

        state_dict = {
            'feature': feature,
            'logit': logit,
            'target': target
        }

        with open(f"{self.logger.log_dir}/data.pkl", "wb") as f:
            pkl.dump(state_dict, f)
            print("file save has been done")

        return outputs


