import os
from os.path import relpath, abspath
from copy import deepcopy
from torch.utils.data import ConcatDataset
from hydra import compose, initialize
from omegaconf import OmegaConf

from .latent import WData
from .attrs import AData
from .attrs2 import AData as AData2

def create_dataset(cfg):
    if cfg.dataset.type == 'w_data':
        dataset = WData(
            cfg=cfg,
            w_path=cfg.dataset.w_path,
            img_path=cfg.dataset.img_path,
            stats_path=cfg.dataset.stats_path,
            padding=cfg.dataset.padding,
            image_size=cfg.dataset.image_size,
            normalize_w=cfg.dataset.normalize_w,
            normalize_image=cfg.dataset.normalize_image,
            w_norm_type=cfg.dataset.w_norm_type,
            z_scaler=cfg.dataset.z_scaler
        )
    elif (cfg.dataset.type == 'a_data') or ((cfg.dataset.type == 'a_data_extended') and (cfg.dataset.mode == 'test')):
        dataset = AData(
            cfg=cfg,
        )
    elif (cfg.dataset.type == 'a_data2') or ((cfg.dataset.type == 'a_data_extended') and (cfg.dataset.mode == 'test')):
        dataset = AData2(
            cfg=cfg,
        )
    elif cfg.dataset.type == 'a_data_extended':
        # save the orig config
        orig_config = deepcopy(cfg)
        base_dataset = AData(
            cfg=cfg,
        )
        
        # We also have to ensure that the the last index is updated
        additional_datasets = []
        for d in cfg.dataset.additional_datasets:
            _curr_cfg = deepcopy(cfg)
            _cfg_d = cfg.dataset
            # load the additional dataset config
            with initialize(config_path=relpath(abspath('../../../configs'), os.getcwd()), version_base='1.2'):
                _new_config = compose('config.yaml', 
                                    overrides=[
                                            f'env={cfg.env.name}',
                                        
                                            f'dataset={d}',
                                            f'dataset.normalize_w={_cfg_d.normalize_w}',
                                            f'dataset.w_norm_type={_cfg_d.w_norm_type}',
                                            f'dataset.z_scaler={_cfg_d.z_scaler}',
                                            f'dataset.padding={_cfg_d.padding}',
                                            ])
                OmegaConf.set_struct(_new_config, True)
            _curr_cfg.dataset = _new_config.dataset
            # overwrite the composer params from the original config
            _curr_cfg.dataset.composer = orig_config.dataset.composer
            additional_datasets.append(AData(cfg=_curr_cfg))
            
        dataset = ConcatDataset([base_dataset] + additional_datasets)
        
    return dataset