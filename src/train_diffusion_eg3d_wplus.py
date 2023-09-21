# train.py
import os
import shutil
import tempfile
import warnings

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from builder import ModelBuilder
from trainer import Trainer


# ------------------------------------------------------------------------------
# Subprocess for training
# ------------------------------------------------------------------------------
def subprocess_fn(rank, c, temp_dir):
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))

    if c.num_gpus > 1:
        init_method = f'file://{temp_dir}/init_file'
        torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Create a model builder
    model_builder = ModelBuilder(c)

    # Initialize the trainer
    trainer = Trainer(c, model_builder)

    # Start the training process
    trainer.train()

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
    
@hydra.main(config_path="..", config_name="experiment_config.yaml")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, True)



if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter