import hydra
from omegaconf import DictConfig, OmegaConf

from .denoising_diffusion_pytorch_1d import GaussianDiffusion1D


def get_diffusion(cfg: DictConfig, model):
    _diffusion = GaussianDiffusion1D(
        model=model,
        seq_length=cfg.diffusion.seq_length+cfg.conditioner.seq_length,
        timesteps=cfg.diffusion.timesteps,
        sampling_timesteps=cfg.diffusion.sampling_timesteps,
        objective =cfg.diffusion.objective,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_kwargs=cfg.diffusion.beta_kwargs,
        ddim_sampling_eta=cfg.diffusion.ddim_sampling_eta,
        loss_type=cfg.diffusion.loss_type,
        loss_kwargs=cfg.diffusion.loss_kwargs,
        auto_normalize=cfg.diffusion.auto_normalize,
        is_self_denoising=cfg.diffusion.is_self_denoising,
    )

    if cfg.diffusion.return_latents:
        _diffusion.forward = _diffusion.forward_w_latents
    else:
        _diffusion.forward = _diffusion.forward_wo_latents

    return _diffusion