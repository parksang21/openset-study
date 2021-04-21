from typing import Any
import pytorch_lightning as pl
import torch.nn

from data.cifar10 import CIFAR10, general_transform

from torch.utils.data import DataLoader
from model import resnet18


class BasePL(pl.LightningModule):
    def __init__(self, lr, batch_size, num_workers, **kwargs):
        super(BasePL, self).__init__()
        self.model = resnet18(with_fc=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_accuracy = pl.metrics.Accuracy()

        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--lr", type=float, default=1e-2)
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=2)
        return parser

    def train_dataloader(self) -> Any:
        cifar10 = CIFAR10('/datasets', train=True, transform=general_transform['train'])
        loader = DataLoader(cifar10, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers)
        return loader

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.model(x)
        loss = self.criterion(logits, y)
        self.log('loss', loss, logger=True)

        _, y_pred = logits.max(1)

        train_acc_batch = self.train_accuracy(y_pred, y)
        self.log('train_acc_batch', train_acc_batch, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=self.lr)

    def training_epoch_end(self, outputs):
        pass
