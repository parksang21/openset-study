import torch
import torch.nn as nn


class SimModel(nn.Module):
    def __init__(self,
                 backbone,
                 fe_size,
                 lat_dim,
                 class_num,
                 hidden_dim=256):
        """Model with self-supervised term

        Args:
            backbone (nn.Module): base feature extractor module
            fe_size (integer): feature extracted vector size
            lat_dim (integer): output latent vector dim
            class_num (int): class number for classifier
            hidden_dim (int, optional): hidden dim of MLP. Defaults to 256.
        """
        super(SimModel, self).__init__()
        self.backbone = backbone
        self.fe_size = fe_size
        self.class_num = class_num
        self.latent_dim = lat_dim
        self.hidden_dim = hidden_dim

        self.g = nn.Sequential(
            nn.Linear(self.fe_size, self.hidden_dim, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.latent_dim, bias=True)
        )

        self.fc = nn.Linear(self.fe_size, self.class_num)

    def forward(self, input):
        """inference

        Args:
            input (Tensor, B * N * D): Tensor consisted of B(batch size)
                                       N (number of instances)
                                       D (feature Dimension)
        Returns:
            similarity vector, fc logit outputs
        """
        embeddings = self.backbone(input)
        sim_out = self.g(embeddings)
        logits = self.fc(embeddings)

        return sim_out, logits