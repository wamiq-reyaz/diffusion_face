import os
import numpy as np
import torch
import torch.optim as optim
from einops.layers.torch import Rearrange

from .attr_conditional import Trainer as OriginalTrainer
from .attr_conditional import FixedPositionalEncoding


class MLPEmbedding(torch.nn.Module):
    def __init__(self,
                proj_dims,
                hidden_dims,
                out_dims,
                num_layers,
                dropout=0.0,
                activation='ReLU'):
        super().__init__()
        self.proj_dims = proj_dims
        self.hidden_dims = hidden_dims
        self.out_dims = out_dims
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = getattr(torch.nn, activation)

        self.layers = torch.nn.ModuleList()
        self.layers.append(FixedPositionalEncoding(proj_dims=proj_dims))
        self.layers.append(Rearrange('b e s -> b s e'))
        for _ in range(num_layers):
            self.layers.append(torch.nn.Linear(proj_dims, hidden_dims))
            self.layers.append(self.activation())
            self.layers.append(torch.nn.Dropout(dropout))
            proj_dims = hidden_dims
        self.layers.append(torch.nn.Linear(proj_dims, out_dims))
        self.layers.append(Rearrange('b s e -> b e s'))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Trainer(OriginalTrainer):
    def __init__(self,
                cfg,
                model_builder,
                rank):
        super().__init__(cfg, model_builder, rank)
        
        # overwrite the embedding function
        self.embedder = torch.nn.parallel.DistributedDataParallel(
            MLPEmbedding(
                proj_dims=self.cfg.training.proj_dims,
                hidden_dims=self.cfg.training.hidden_dims,
                out_dims=self.cfg.training.out_dims,
                num_layers=self.cfg.training.num_layers,
                dropout=self.cfg.training.dropout,
                activation=self.cfg.training.activation
            ).cuda(),
            device_ids=[rank],
            broadcast_buffers=True
        )
    
        # reinit the optimizer and scheduler
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

    

