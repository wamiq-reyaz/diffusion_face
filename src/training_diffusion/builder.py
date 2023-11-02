import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from itertools import cycle

from .diffusion.builder import get_diffusion
from .models.builder import get_model
from .datasets.builder import create_dataset
from .conditioners.builder import get_conditioner

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from copy import deepcopy

class InfiniteDataSampler(DistributedSampler):
    def __init__(self,
                 dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.epoch = 0

    def __iter__(self):
        while True:
            self.epoch += 1
            super().set_epoch(self.epoch)
            yield from super().__iter__()
        

class ModelBuilder:
    def __init__(self, cfg: DictConfig, rank: int = 0):
        self.cfg = cfg

        # Create a dataset
        self.dataset = create_dataset(cfg)
        if cfg.num_gpus > 1:
            sampler = InfiniteDataSampler(self.dataset,
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
                                                        shuffle=False,
                                                        drop_last=True)
        # Make infinite. WARNING: shuffle is performed only once
        # self.train_loader = cycle(self.train_loader)

        # if there exist some special test settings, use them
        if hasattr(cfg.dataset, 'mode'):
            test_cfg = deepcopy(cfg)
            test_cfg.dataset.mode = 'test'
            test_cfg.dataset.idxes = [140000, 140000+1024]
        self.test_dataset = create_dataset(test_cfg)
        self.test_dataset = Subset(self.test_dataset, range(1024))

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=cfg.training.test_batch_gpu,
                                                        num_workers=2,
                                                        sampler=None,
                                                        pin_memory=True,
                                                        shuffle=False,
                                                        drop_last=True)

        # Create a model
        model = get_model(cfg)
        diffusion = get_diffusion(cfg, model)
        self.model = diffusion.cuda()
        if cfg.num_gpus > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[rank],
                                                                   broadcast_buffers=True)

        # Create a conditioner
        self.conditioner = get_conditioner(cfg)
        if self.conditioner:
            self.conditioner = self.conditioner.cuda()
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
        return self.test_loader # TODO