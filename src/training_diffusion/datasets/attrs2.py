import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from .utils import (
    random_brush, random_mask, mask_category, get_valid_attrib_idx,
    compute_idxes
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
ATTR_REPLACEMENT_VALUE = -1.0   
SEG_REPLACEMENT_VALUE = -1.0 / 19
RGB_REPLACEMENT_VALUE = 0.0

class AData(Dataset):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        self.cfg = cfg
        cfg_d = cfg.dataset
        self.w_path = cfg_d.w_path
        self.img_path = cfg_d.img_path
        self.attr_path = cfg_d.attr_path
        self.seg_path = cfg_d.seg_path
        self.stats_path = cfg_d.stats_path
        self.normalize_w = cfg_d.normalize_w
        self.normalize_image = cfg_d.normalize_image
        self.w_norm_type = cfg_d.w_norm_type
        self.z_scaler = cfg_d.z_scaler
        self.mode = cfg_d.mode # train or val. Controls the masks and the dropout.
        
        # ----------------------------
        # Composer dropout and masking params
        # ----------------------------
        cfg_c = cfg.dataset.composer
        self.all_retain_prob = cfg_c.all_retain_prob # probability of retaining all attributes and RGB
        self.attr_dropout = cfg_c.attr_dropout # probability of dropping an attribute
        self.attr_retain_prob = cfg_c.attr_retain_prob # probability of retaining all attributes
        self.seg_retain_prob = cfg_c.seg_retain_prob # probability of retaining the segmentation condition
        self.rgb_retain_prob = cfg_c.rgb_retain_prob # probability of retaining the RGB condition
        self.mask_types_rgb = cfg_c.mask_types_rgb
        self.mask_types_seg = cfg_c.mask_types_seg
        self.hole_range = cfg_c.hole_range
        self.min_num_vertex = cfg_c.min_num_vertex
        self.max_num_vertex = cfg_c.max_num_vertex
        self.mean_angle = cfg_c.mean_angle
        self.angle_range = cfg_c.angle_range
        self.min_width = cfg_c.min_width
        self.max_width = cfg_c.max_width
        self.average_radius = cfg_c.average_radius
        self.hole_range = cfg_c.hole_range
        self.n_classes = cfg_c.n_classes
        self.max_tries_brush = cfg_c.max_tries_brush
        self.max_tries_class = cfg_c.max_tries_class # number of classes to drop
        self.drop_seg = cfg_c.drop_seg # boolean, whether to drop the segmentation condition
        self.drop_rgb = cfg_c.drop_rgb # boolean, whether to drop the RGB condition
        
        self.drop_rgb_prob = cfg_c.drop_rgb_prob
        self.drop_seg_prob = cfg_c.drop_seg_prob
        
        self.jitter_prob = cfg_c.jitter_prob
        self.jitter_max = cfg_c.jitter_max
        self.jitter_rot_max = cfg_c.jitter_rot_max
        
        # ----------------------------
        # Padding and image size
        # ----------------------------
        _p = [int(p) for p in cfg_d.padding]
        self.padding = _p
        if isinstance(cfg_d.image_size, int):
            image_size = (cfg_d.image_size, cfg_d.image_size)
        self.image_size = image_size
        
        # ----------------------------
        # Normalization
        # ----------------------------
        stats = torch.load(self.stats_path)
        
        if self.normalize_w:
            _min, _range = stats['min'].numpy(), stats['range'].numpy()
            # _range = _max - _min
            _range[_range == 0] = 1.
            self._min = torch.from_numpy(_min)
            self._range = torch.from_numpy(_range)

            if self.w_norm_type == 'z_norm':
                self._mean = torch.from_numpy(stats['mean'].numpy())
                self._std = torch.from_numpy(stats['std'].numpy())
                self._std[self._std == 0] = 1.
                # z_scaler ensures that returned values are in the range of [-1, 1]
                self._std = self._std * self.z_scaler
                
        # ----------------------------
        # Getting indices from the dataset for corrupt data
        # ----------------------------
        self.idxes = compute_idxes(cfg_d.idxes)
        
    def __len__(self):
        return len(self.idxes)
    
    def __getitem__(self, idx):
        idx = self.idxes[idx]
        _name = str(idx).zfill(7)
        if not self._all_exist(_name):
            _name = '0000000'
        
        # ----------------------------
        # Load data
        # ----------------------------
        w = np.load(os.path.join(self.w_path, _name + '.npy'))
        img_pil = Image.open(os.path.join(self.img_path, _name + '.png'))
        attr = np.load(os.path.join(self.attr_path, _name + '.npy'))
        seg_pil = Image.open(os.path.join(self.seg_path, _name + '.png'))
        
        # ----------------------------
        # Normalize data
        # ----------------------------
        w = torch.from_numpy(w).float() # SxE
        if self.normalize_w:
            if self.w_norm_type == 'min_max':
                w = (w - self._min) / self._range
            elif self.w_norm_type == 'z_norm':
                w = (w - self._mean) / self._std
            else:
                raise ValueError('Unknown w_norm_type: {}'.format(self.w_norm_type))
            
        img = img_pil.resize(
            self.image_size,
            resample=Image.Resampling.BILINEAR
        )
        img = np.asarray(img).astype(np.float32) / 255.0 # HxWx3
        img = torch.from_numpy(img).permute(2,0,1) # 3xHxW
        
        if self.normalize_image:
            img = (img - IMAGENET_MEAN[:,None,None]) / IMAGENET_STD[:,None,None]
            
        seg_pil = seg_pil.resize(
            self.image_size,
            resample=Image.Resampling.NEAREST
        )
        # perform jittering and rotation on the segmentation only
        if torch.rand(1).item() < self.jitter_prob:
            jitter = torch.randint(-self.jitter_max, self.jitter_max, (2,))
            seg_pil = seg_pil.transform(
                seg_pil.size,
                Image.AFFINE,
                (1, 0, jitter[0], 0, 1, jitter[1]),
                Image.NEAREST
            )
            jitter_rot = -self.jitter_rot_max + (torch.rand((1,)) * self.jitter_rot_max, (1,))
            seg_pil = seg_pil.rotate(jitter_rot)
        
        seg = np.asarray(seg_pil).astype(np.float32) / self.n_classes # HxW 
        seg = torch.from_numpy(seg).unsqueeze(0) # 1xHxW
        
        attr = self._get_relevant_attr(attr)
        attr = torch.from_numpy(attr).float() # N
        # ----------------------------
        # Generate masks and introduce dropout
        # ----------------------------
        processed_data = self._mask_data({'w': w, 'img': img, 'attr': attr, 'seg': seg})
        
        # ----------------------------
        # Final touches - close image and pad
        # ----------------------------
        img_pil.close()
        if self.padding[0] + self.padding[1] > 0:
            processed_data['w'] = torch.nn.functional.pad(processed_data['w'], (0, 0, self.padding[0], self.padding[1]), mode='constant', value=0)
        processed_data['w'] = processed_data['w'].permute(1,0) # ExS
            
        return {'data': processed_data['w'],
                'condition': torch.cat(
                                    [processed_data['img'],
                                    processed_data['seg'],
                                    ]),
                'attr': processed_data['attr'],
                'img_mask': processed_data['img_mask'],
                'seg_mask': processed_data['seg_mask'],
                'attr_mask':processed_data['attr_mask'],
                'idx': torch.tensor(idx, dtype=torch.int32).view(1),
                'name': _name}
        
    def _all_exist(self, _name):
        if os.path.exists(os.path.join(self.w_path, _name + '.npy')) and \
            os.path.exists(os.path.join(self.img_path, _name + '.png')) and \
            os.path.exists(os.path.join(self.attr_path, _name + '.npy')) and \
            os.path.exists(os.path.join(self.seg_path, _name + '.png')):
            return True
        else:
            return False
        
    def _mask_data(self, kwargs):
        ''' kwargs: w, img, attr, seg
            Using the previously set arguments, generate masks and dropout conditions
        '''
        w = kwargs['w']
        img = kwargs['img']
        attr = kwargs['attr']
        seg = kwargs['seg']

        if self.drop_rgb:
            img = torch.ones_like(img) * RGB_REPLACEMENT_VALUE
        if self.drop_seg:
            seg = torch.ones_like(seg) * SEG_REPLACEMENT_VALUE
            
        # return early
        if self.mode in ['val', 'test']:
            img_mask = torch.ones(1)
            seg_mask = torch.ones(1)
            attr_mask = torch.ones(attr.shape)
            return {'w': w,
                    'img': img,
                    'attr': attr,
                    'seg': seg,
                    'img_mask': img_mask,
                    'seg_mask': seg_mask,
                    'attr_mask': attr_mask
                    }
        else:
            # first handle masks and then we handle dropout
            _mask_seg, _mask_rgb = self._gen_mask(seg)
            _mask_rgb = _mask_rgb.repeat(3,1,1).bool()
            _mask_seg = _mask_seg.repeat(1,1,1).bool()
            
            # apply masks
            img[~_mask_rgb] = RGB_REPLACEMENT_VALUE
            seg[~_mask_seg] = SEG_REPLACEMENT_VALUE
            
            # apply whole masks
            if torch.rand(1).item() < self.drop_rgb_prob:
                img = torch.ones_like(img) * RGB_REPLACEMENT_VALUE
            if torch.rand(1).item() < self.drop_seg_prob:
                seg = torch.ones_like(seg) * SEG_REPLACEMENT_VALUE
            
            if torch.rand(1).item() < self.all_retain_prob:
                return {'w': w,
                        'img': img,
                        'attr': attr,
                        'seg': seg,
                        'img_mask': torch.ones(1),
                        'seg_mask': torch.ones(1),
                        'attr_mask': torch.ones(attr.shape)
                        }
            
            if (torch.rand(1).item() > self.attr_retain_prob):
                _shape = attr.shape
                attr_mask = torch.rand(_shape) < self.attr_dropout
                attr = attr * (attr_mask  * ATTR_REPLACEMENT_VALUE)
                attr = attr
            else:
                attr_mask = torch.ones(attr.shape)
                
            if torch.rand(1).item() > self.seg_retain_prob:
                seg_mask = torch.zeros(1)
            else:
                seg_mask = torch.ones(1)
            
            if torch.rand(1).item() > self.rgb_retain_prob:
                img_mask = torch.zeros(1)
            else:
                img_mask = torch.ones(1)
                
            return {'w': w.float(),
                    'img': img.float(),
                    'attr': attr.float(),
                    'seg': seg.float(),
                    'img_mask': img_mask.float(),
                    'seg_mask': seg_mask.float(),
                    'attr_mask': attr_mask.float()
                    }
            
    def _get_relevant_attr(self, attr):
        ''' attr: (1xN), torch.float32
            return: (1xK), torch.float32
        '''
        return attr[get_valid_attrib_idx()]
            
    def _gen_mask(self, seg):
        ''' Generate masks for segmentation and RGB images from the given parameters
            return: _mask_seg, _mask_rgb
                    _mask_seg: (1xHxW), torch.float32 segmentation mask
                    _mask_rgb: (1xHxW), torch.float32 RGB mask
        '''
        _rgb_masks = []
        _seg_masks = []
        if self.mask_types_rgb[0] == 'None':
            _rgb_masks.append(torch.ones(1, **self.image_size).float())
        if self.mask_types_seg[0] == 'None':
            _seg_masks.append(torch.ones(1, **self.image_size).float())

        if 'brush' in self.mask_types_rgb:
            _mask = random_brush(max_tries=self.max_tries_brush,
                                s=self.image_size[0],
                                min_num_vertex=self.min_num_vertex,
                                max_num_vertex=self.max_num_vertex,
                                mean_angle=self.mean_angle,
                                angle_range=self.angle_range,
                                min_width=self.min_width,
                                max_width=self.max_width,
                                average_radius=self.average_radius)
            _rgb_masks.append(torch.from_numpy(_mask))
        if 'box' in self.mask_types_rgb:
            _mask = random_mask(s=self.image_size[0],
                                hole_range=self.hole_range)
            _rgb_masks.append(torch.from_numpy(_mask))
        if 'class' in self.mask_types_rgb:
            _mask = mask_category(max_tries=self.max_tries_class,
                                n_classes=self.n_classes,
                                img=seg.numpy())
            _rgb_masks.append(torch.from_numpy(_mask))
            
        if 'brush' in self.mask_types_seg:
            _mask = random_brush(max_tries=self.max_tries_brush,
                                s=self.image_size[0],
                                min_num_vertex=self.min_num_vertex,
                                max_num_vertex=self.max_num_vertex,
                                mean_angle=self.mean_angle,
                                angle_range=self.angle_range,
                                min_width=self.min_width,
                                max_width=self.max_width,
                                average_radius=self.average_radius)
            _seg_masks.append(torch.from_numpy(_mask))
        if 'box' in self.mask_types_seg:
            _mask = random_mask(s=self.image_size[0],
                                hole_range=self.hole_range)
            _seg_masks.append(torch.from_numpy(_mask))
        if 'class' in self.mask_types_seg:
            _mask = mask_category(max_tries=self.max_tries_class,
                                n_classes=self.n_classes,
                                img=seg.numpy())
            _seg_masks.append(torch.from_numpy(_mask))            

        _mask_seg = torch.stack(_seg_masks, dim=0).prod(dim=0)
        _mask_rgb = torch.stack(_rgb_masks, dim=0).prod(dim=0)            
        
        return _mask_seg, _mask_rgb