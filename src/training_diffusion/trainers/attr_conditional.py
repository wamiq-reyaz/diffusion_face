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
from ..common import (
                    create_network_from_pkl,
                    generate_images,
                    get_lookat_cam,
                    get_vertices,
                    set_replacement_hook,
                    Unnormalizer,
                    to_uint8,
                    batch_to_PIL
                    )

class FixedPositionalEncoding(torch.nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims//2
        exb = 2 * torch.linspace(0, ll-1, ll) / proj_dims
        self.sigma = 1.0 / torch.pow(val, exb).view(1, -1, 1)
        self.sigma = 2 * torch.pi * self.sigma

    def forward(self, x):
        ''' x: BxS
            returns BxExS
        '''
        return torch.cat([
            torch.sin(x.unsqueeze(1) * self.sigma.to(x.device)),
            torch.cos(x.unsqueeze(1) * self.sigma.to(x.device))
        ], dim=1)

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
        # create a sin_cos embedder for the attributes
        self.embedder = FixedPositionalEncoding(proj_dims=self.cfg.model.channels)

        # resuming
        if self.cfg.training.resume == "latest":
            self.load_ckpt(ckpt_type="latest")
        elif self.cfg.training.resume == "best":
            self.load_ckpt(ckpt_type="best")
        else:
            pass
                
        # metrics
        self.metric_criterion = 'mse'

        # TODO: remove all hardcoding to inherit from config
        self.stylegan = create_network_from_pkl(self.cfg.training.stylegan_path,
                                                device='cpu',
                                                reload_modules=False,
                                                verbose=False)
        self.unnormalizer = Unnormalizer(path=self.cfg.dataset.stats_path,
                                        mode=self.cfg.dataset.w_norm_type)
        self.verts = get_vertices(self.cfg.dataset.obj_path,
                                self.cfg.dataset.lms_path,
                                self.cfg.dataset.lms_cond,
                                device='cpu')
        self.cams = get_lookat_cam()
        self.gen_fn = generate_images


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
        condition = self.conditioner(batch['condition'].cuda()) # BxExS
        # TODO: separate out the RGB and the segmentation models
        seg_mask = batch['seg_mask'].cuda().squeeze() # B
        # mask out the condition
        condition = condition * seg_mask[:, None, None]  
        attr_condition = batch['attr'].cuda() # BxS'
        attr_condition = self.embedder(attr_condition) # BxExS'
        attr_mask = batch['attr_mask'].cuda() # BxS'
        attr_condition = attr_condition * attr_mask[:, None, :] # BxExS'
        
        condition = torch.cat([condition, attr_condition], dim=2) # BxEx(S+S')
        
        loss = self.model(batch['data'].cuda(), condition=condition) 
        for k, v in loss.items():
            loss[k].backward()
        # perform gradient clipping
        if self.cfg.training.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.conditioner.parameters(), self.cfg.training.clip_grad_norm)
        if (train_step % self.cfg.training.gradient_accumulation_steps == 0) and train_step > 0:
            self.optimizer.step()
        return loss
        
    def validate_on_batch(self, batch, val_step):
        orig_img = batch['condition'].cuda().clone().detach()
        condition = self.conditioner(batch['condition'].cuda())
        attr_condition = batch['attr'].cuda() # BxS'
        attr_condition = self.embedder(attr_condition) # BxExS'
        condition = torch.cat([condition, attr_condition], dim=2) # BxEx(S+S')

        if self.cfg.num_gpus > 1:
            m = self.model.module
        else:
            m = self.model

        m = self.ema.ema_model if self.ema else m

        # We clip the latents only if we are using the auto-normalized diffusion
        latents = m.ddim_sample(batch['data'].shape,
                                condition=condition,
                                clip_denoised=True)
        mse = torch.mean((latents - batch['data'].cuda())**2)
        images = self.gen_fn(G=self.stylegan,
                            w=latents.clone().detach().cpu()[0:1].permute(0,2,1),
                            v=self.verts,
                            camera_params=self.cams,
                            batched=True,
                            unnormalizer=self.unnormalizer,
                            device='cpu',
                            )
        images = torch.nn.functional.interpolate(images, size=(256,256))
        images = to_uint8(images)

        # denormalize the condition images with imagenet stats
        condition = orig_img[0:1].clone().detach()
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).numpy()
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).numpy()
        condition = condition * imagenet_std + imagenet_mean
        condition = (condition * 255).permute(0,2,3,1).numpy()

        concat_images = np.concatenate([condition, images.cpu().numpy()], axis=1)
        concat_images = concat_images.squeeze()
            
        return {'mse': mse}, {'mse': mse}, concat_images

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
    
    
