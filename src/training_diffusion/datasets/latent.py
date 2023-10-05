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

        _p = [int(p) for p in padding]
        self.padding = _p
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        stats = torch.load(self.stats_path)

        if self.normalize_w:
            _min, _range = stats['min'].numpy(), stats['range'].numpy()
            # _range = _max - _min
            _range[_range == 0] = 1.
            self._min = torch.from_numpy(_min)
            self._range = torch.from_numpy(_range)

        total = 2000000 - 128
        per_gpu = total // 4
        completed = 210000
        starts = [0, 499968, 999936, 1499904]
        ends = [s+completed for s in starts]

        self.idxes = []
        for s,e in zip(starts, ends):
            self.idxes += list(range(s,e))


    def __len__(self):
        return len(self.idxes) # TODO: change to accept this from cfg

    def __getitem__(self, idx):
        idx = self.idxes[idx]
        _name = str(idx).zfill(7)
        if not os.path.exists(os.path.join(self.w_path, _name + '.npy')) or \
            not os.path.exists(os.path.join(self.img_path, _name + '.png')):
            _name = '0000000'
        data = np.load(os.path.join(self.w_path, _name + '.npy'))
        data = torch.from_numpy(data).float() # SxE
        if self.normalize_w:
            data = data - self._min
            data = data / self._range

        if self.padding[0] + self.padding[1] > 0:
            data = torch.nn.functional.pad(data, (0, 0, self.padding[0], self.padding[1]), mode='constant', value=0)

        data = data.permute(1,0) # ExS

        # ----------------------------------------

        img_pil = Image.open(
            os.path.join(
                self.img_path,
                _name + '.png'
            )
        )
        img = img_pil.resize(
            self.image_size,
            resample=Image.Resampling.BILINEAR
        )
        img = np.array(img).astype(np.float32) / 255.0 # HxWx3
        img = torch.from_numpy(img).permute(2,0,1) # 3xHxW

        if self.normalize_image:
            img = img - IMAGENET_MEAN[:, None, None]
            img = img / IMAGENET_STD[:, None, None]
        
        # close the image
        img_pil.close()

        return {'data': data,
                'condition': img,
                'idx': torch.tensor(idx, dtype=torch.int32).view(1),
                'name': _name}



