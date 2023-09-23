import os
import sys
import pickle
from typing import Any, List, Dict, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

class WData(Dataset):
    def __init__(self,
                 cfg,
                w_path,
                img_path,
                stats_path,
                image_size=(256,256),
                padding=(0,0),
                normalize_w=True,
                normalize_image=True,):

        self.cfg = cfg
        self.w_path = w_path
        self.img_path = img_path
        self.stats_path = stats_path
        self.normalize_w = normalize_w
        self.normalize_image = normalize_image
        self.padding = padding
        self.image_size = image_size

        stats = torch.load(self.stats_path)

        if self.normalize:
            _min, _max = stats['min'].numpy(), stats['max'].numpy()
            _range = _max - _min
            _range[_range == 0] = 1.
            self._min = torch.from_numpy(_min)
            self._range = torch.from_numpy(_range)

    def __len__(self):
        return 500000 # TODO: change to accept this from cfg

    def __getitem__(self, idx):
        _name = str(idx).zfill(7)
        data = np.load(os.path.join(self.w_path, _name + '.npy'))
        data = torch.from_numpy(data).float() # SxE
        if self.normalize_w:
            data = data - self._min
            data = data / self._range

        if self.padding[0] + self.padding[1] > 0:
            data = torch.nn.functional.pad(data, (0, 0, self.padding[0], self.padding[1]), mode='constant', value=0)

        data = data.permute(1,0) # ExS

        # ----------------------------------------

        img = Image.open(
            os.path.join(
                self.img_path,
                _name + '.png'
            )
        )
        img = img.resize(
            self.image_size,
            resample=Image.Resampling.BILINEAR
        )
        img = np.array(img).astype(np.float32) / 255.0 # HxWx3
        img = torch.from_numpy(img).permute(1,2,0) # 3xHxW

        if self.normalize_image:
            img = img - IMAGENET_MEAN[:, None, None]
            img = img / IMAGENET_STD[:, None, None]

        return {'data': data,
                'condition': img}



