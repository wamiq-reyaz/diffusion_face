import os
import uuid

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import wandb
from tqdm import tqdm

from .base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self,
                 cfg,
                 model_builder,
                 rank):
        super().__init__(cfg, model_builder, rank)
        self.conditioner = model_builder.get_conditioner()

        # Redo the optimizer and scheduler to include the conditioner
        self.optimizer = self.init_optimizer()
        self.scheduler = self.init_scheduler()

        # resuming
        if self.cfg.training.resume == "latest":
            self.load_ckpt(ckpt_type="latest")
        elif self.cfg.training.resume == "best":
            self.load_ckpt(ckpt_type="best")
        else:
            pass
                
        # metrics
        self.metric_criterion = 'mse'

    def load_ckpt(self, checkpoint_dir="./checkpoints", ckpt_type="best"):
        _dir = os.path.join(self.cfg.experiment_dir, checkpoint_dir)
        if ckpt_type == "best":
            ckpt_path = os.path.join(checkpoint_dir, "best.ckpt")
        elif ckpt_type == "latest":
            ckpt_path = os.path.join(checkpoint_dir, "latest.ckpt")
        else:
            raise ValueError(f"ckpt_type {ckpt_type} not supported")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if self.cfg.num_gpus > 1:
            self.model.module.load_state_dict(ckpt["model"])
            self.conditioner.module.load_state_dict(ckpt["conditioner"])
        else:
            self.model.load_state_dict(ckpt["model"])
            self.conditioner.load_state_dict(ckpt["conditioner"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]

    def init_optimizer(self):
        m = self.model.module if self.cfg.num_gpus > 1 else self.model
        cm = self.conditioner.module if self.cfg.num_gpus > 1 else self.conditioner

        param_groups = []
        if cm is not None:
            param_groups.append({'params': cm.parameters(),
                                'lr': self.cfg.optimizer.lr2,
                                'weight_decay': self.cfg.optimizer.wd2,
                                'betas': self.cfg.optimizer.betas2,
                                'eps': self.cfg.optimizer.eps2})
            
        param_groups.append({'params': m.parameters(),
                                'lr': self.cfg.optimizer.lr,
                                'weight_decay': self.cfg.optimizer.wd,
                                'betas': self.cfg.optimizer.betas,
                                'eps': self.cfg.optimizer.eps})
        
        if self.cfg.optimizer.name == 'adam':
            optimizer = optim.Adam(param_groups)
        elif self.cfg.optimizer.name == 'adamw':
            optimizer = optim.AdamW(param_groups)

        return optimizer

    def train_on_batch(self, batch, train_step):
        self.optimizer.zero_grad()
        condition = self.conditioner(batch['condition'])
        loss = self.model(batch['data'], condition=condition) 
        for k, v in loss.items():
            loss[k].backward()
        # perform gradient clipping
        if self.cfg.training.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.conditioner.parameters(), self.cfg.training.clip_grad_norm)
        self.optimizer.step()
        return loss
        
    def validate_on_batch(self, batch, val_step):
        condition = self.conditioner(batch['condition'])
        if self.cfg.num_gpus > 1:
            m = self.model.module
        else:
            m = self.model

        m = self.ema if self.ema else m

        latents = m.ddim_sample(batch['data'].shape, condition=condition)
        mse = torch.mean((latents - batch['data'].cuda())**2)
        return {'mse': mse}, {'mse': mse}

    def save_checkpoint(self, filename):
        if not self.should_write:
            return
        root = os.path.join(self.cfg.experiment_dir, 'checkpoints')
        os.makedirs(root, exist_ok=True)

        fpath = os.path.join(root, filename)
        if self.cfg.num_gpus > 1:
            m = self.model.module
            cm = self.conditioner.module
        else:
            m = self.model
            cm = self.conditioner

        torch.save({
            'model': m.state_dict(),
            'conditioner': cm.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema else None,
            'step': self.step,
        }, fpath)
    
    
