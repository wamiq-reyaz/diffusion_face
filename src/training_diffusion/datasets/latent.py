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
                w_path,
                img_path,
                stats_path,
                image_size=(256,256),
                padding=(0,0),
                normalize_w=True,
                normalize_image=True):

        self.w_path = w_path
        self.img_path = img_path
        self.stats_path = stats_path
        self.normalize_w = normalize_w
        self.normalize_image = normalize_image
        self.padding = padding
        self.image_size = image_size

        self.data = torch.load(self.w_path).numpy()
        stats = torch.load(self.stats_path)
        print('Loaded Data')

        if self.normalize:
            _min, _max = stats['min'].numpy(), stats['max'].numpy()
            _range = _max - _min
            _range[_range == 0] = 1.

            self.data = self.data - _min[np.newaxis, :]
            self.data = self.data / _range[np.newaxis, :]
        
        self.data = torch.from_numpy(self.data)

        if self.padding != (0, 0):
            self.data = torch.nn.functional.pad(self, self.data, pad=(0, 0, self.pading[0], self.padding[1]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data = self.data[idx, :].permute(1,0) # SxE

        img = Image.open(
            os.path.join(
                self.img_path,
                str(idx).zfill(7) + '.png'
            )
        )
        img = img.resize(
            self.image_size,
            resample=Image.Resampling.LANCZOS
        )
        img = np.array(img).astype(np.float32) / 255.0 # HxWx3
        img = torch.from_numpy(img).permute(1,2,0) # 3xHxW

        if self.normalize_image:
            img = img - IMAGENET_MEAN[:, None, None]
            img = img / IMAGENET_STD[:, None, None]

        return {'data': data,
                'condition': img}



def create_dataset(opts):
    dataset = WData(w_path=opts['w_path'],
                    img_path=opts['img_path'],
                    stats_path=opts['stats_path'],
                    padding=opts['padding'],
                    image_size=opts['image_size'],
                    normalize_w=opts['normalize_w'],
                    normalize_image=opts['normalize_image'])

    return dataset