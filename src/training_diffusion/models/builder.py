import hydra
from omegaconf import DictConfig, OmegaConf


from .unet import Unet1D
from .uvit import UViT


def get_model(cfg: DictConfig):
    if cfg.model.name == 'unet':
        return Unet1D(  cfg=cfg,
                        dim=cfg.model.dim,
                        channels=cfg.model.channels,
                        dim_mults=cfg.model.dim_mults,
                        out_dim=cfg.model.out_dim,
                        is_conditional=cfg.model.is_conditional,
                        add_condition=cfg.model.add_condition,
                        resnet_block_groups=cfg.model.resnet_block_groups,
                        learned_variance=cfg.model.learned_variance,
                        learned_sinusoidal_cond=cfg.model.learned_sinusoidal_cond,
                        random_fourier_features=cfg.model.random_fourier_features,
                        learned_sinusoidal_dim=cfg.model.learned_sinusoidal_dim,
                        scale_condition=cfg.model.scale_condition,
                        )
    elif cfg.model.name == 'uvit':
        return UViT( cfg=cfg,
                    extras=cfg.model.extras,
                    len_latents=cfg.model.len_latents,
                    embed_dim=cfg.model.embed_dim,
                    channels=cfg.model.channels,
                    depth=cfg.model.depth,
                    num_heads=cfg.model.num_heads,
                    mlp_ratio=cfg.model.mlp_ratio,
                    qkv_bias=cfg.model.qkv_bias,
                    qk_scale=cfg.model.qk_scale,
                    norm_layer=cfg.model.norm_layer,
                    mlp_time_embed=cfg.model.mlp_time_embed,
                    use_checkpoint=cfg.model.use_checkpoint,
                    skip=cfg.model.skip,
                    scale_y=cfg.model.scale_y,)
    else:
        raise ValueError(f'Unknown model: {cfg.model.name}')