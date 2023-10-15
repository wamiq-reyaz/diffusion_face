# train.py
import os
import shutil
import tempfile
import warnings
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from infra.utils import diffusion_length_resolver
OmegaConf.register_new_resolver(name='diffusion_length_resolver',
                                resolver=diffusion_length_resolver)
import torch

from training_diffusion.builder import ModelBuilder
from training_diffusion.trainers.builder import get_trainer
import wandb


# ------------------------------------------------------------------------------
# Subprocess for training
# ------------------------------------------------------------------------------
def subprocess_fn(rank, cfg, temp_dir):
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))

    if cfg.num_gpus > 1:
        init_method = f'file://{temp_dir}/init_file'
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=cfg.num_gpus)

    # ----------------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------------
    torch.cuda.set_device(rank)

    # Create a model builder
    model_builder = ModelBuilder(cfg=cfg, rank=rank)

    # Initialize the trainer
    trainer = get_trainer(cfg, model_builder, rank=rank)

    try:
        # Start the training process
        trainer.train()
    except Exception as e:
        raise e
    finally:
        wandb.finish()


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
    
@hydra.main(config_path="..", config_name="experiment_config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, True)
    torch.multiprocessing.set_start_method('spawn')

    # ----------------------------------------------------------------------------
    # Seed everything
    # ----------------------------------------------------------------------------
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    random.seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Create a temporary directory for distributed training
    with tempfile.TemporaryDirectory() as temp_dir:
        if cfg.num_gpus == 1:
            subprocess_fn(rank=0, cfg=cfg, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(cfg, temp_dir), nprocs=cfg.num_gpus)



if __name__ == "__main__":
    
    main() # pylint: disable=no-value-for-parameter