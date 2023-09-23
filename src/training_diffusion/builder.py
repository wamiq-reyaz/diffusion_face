import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from diffusion.builder import get_diffusion
from models.builder import get_model
from datasets.builder import create_dataset
from conditioners import get_conditioner


class ModelBuilder:
    def __init__(self, cfg: DictConfig, rank: int = 0):
        self.cfg = cfg

        # Create a dataset
        self.dataset = create_dataset(cfg)
        if cfg.num_gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                    num_replicas=cfg.num_gpus,
                                                                    rank=rank,
                                                                    shuffle=True)
        else:
            sampler = None

        self.train_loader = torch.utils.data.DataLoader(self.dataset,
                                                        batch_size=cfg.training.per_gpu_batch_size,
                                                        num_workers=cfg.training.workers,
                                                        sampler=sampler,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        drop_last=True)

        # Create a model
        model = get_model(cfg)
        diffusion = get_diffusion(cfg, model)
        self.model = diffusion
        if cfg.num_gpus > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[rank],
                                                                   broadcast_buffers=True)

        # Create a conditioner
        self.conditioner = get_conditioner(cfg)
        if self.conditioner:
            if cfg.num_gpus > 1:
                self.conditioner = torch.nn.parallel.DistributedDataParallel(self.conditioner,
                                                                         device_ids=[rank],
                                                                         broadcast_buffers=True)

    def get_model(self):
        return self.model

    def get_conditioner(self):
        return self.conditioner

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return None # TODO