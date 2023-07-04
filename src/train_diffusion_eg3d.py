import os
import sys
import pickle
from typing import Any
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Trainer1D, GaussianDiffusion1D, Unet1D

import torch
import torch.nn as nn
import tensorboard as tb
from torch.utils.data import Dataset
import numpy as np


class WData(Dataset):
    def __init__(self,
                path,
                normalize=True):
        self.path = path
        self.normalize = normalize

        with open(self.path, 'rb') as fd:
            all_ws = pickle.load(fd)

        all_keys = list(all_ws.keys())
        self.data = np.empty((len(all_keys), 512), dtype=np.float32)

        for ii, kk in enumerate(all_keys):
            self.data[ii, :] = all_ws[kk][0, 0, :] # it is a 1x28x512 w

        if self.normalize:
            mean, std = np.mean(self.data, axis=0), np.std(self.data, axis=0)
            _min, _max = np.min(self.data, axis=0), np.max(self.data, axis=0)
            _range = _max - _min
        
        # self.data = self.data - mean[np.newaxis, :]
        # self.data = self.data / std[np.newaxis, :]

        self.data = self.data - _min[np.newaxis, :]
        self.data = self.data / _range[np.newaxis, :]
        
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: Any) -> Any:
        return self.data[index, :].unsqueeze(0) # 1x512
    

if __name__ == '__main__':
    model = Unet1D(
    dim = 512,
    channels=1,
    dim_mults = (1, 2, 4),
    out_dim = 1
    )

    print(model)

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 512,
        timesteps = 1000,
        objective = 'pred_v'
    )

    dataset = WData(path='/storage/nfs/wamiq/next3d/data/generated_samples/attempt1/all_ws.pkl') # features are normalized from 0 to 1

    # Or using trainer

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32*4,
        train_lr = 8e-5,
        train_num_steps = 20000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.9999,                # exponential moving average decay
        ema_update_every=10,
        amp = False,                       # turn on mixed precision
        results_folder='min_max_demo',
        save_and_sample_every=100
    )
    trainer.train()

    # after a lot of training

    sampled_seq = diffusion.sample(batch_size = 4)
    with open('aa.pkl', 'wb') as fd:
        pickle.dump(sampled_seq, fd)
    sampled_seq.shape # (4, 32, 128)
    


