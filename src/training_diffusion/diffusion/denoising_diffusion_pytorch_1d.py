import math
from multiprocessing import cpu_count
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from contextlib import nullcontext

from tqdm.auto import tqdm

from .version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# new hook type
def set_replacement_hook(generator: torch.nn.Module, names, tensors):
    all_hooks = []
    for ii, name in enumerate(names):
        for modname, module in generator.named_modules():
            if modname == name:
                mod_hook = partial(replace_hook, name, tensors[:, ii, :])
                hook = module.register_forward_hook(mod_hook)
                all_hooks.append(hook)
    
    return all_hooks

# Logging and color utils 
# TODO: move to separate utils file

def colorize(img, cmap='viridis'):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        pass
    else:
        raise TypeError('Only numpy arrays and torch tensors supported')
    
    cmapper = matplotlib.cm.get_cmap(name=cmap)
    img = cmapper(img, bytes=False).astype(np.float32) # returns stuff as HxWx4
    img = (img[..., :3] ) *255
    img = img.astype(np.uint8)

    return img
    

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    scale = 1000 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, beta_s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + beta_s) / (1 + beta_s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        beta_kwargs = {},
        ddim_sampling_eta = 0.,
        loss_type = 'l1',
        loss_kwargs = {},
        auto_normalize = True,
        is_self_denoising=False, # self denoising means that you use you own noise to clean the input

    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length
        self.is_self_denoising = is_self_denoising
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps,  beta_start=beta_kwargs['beta_start'], beta_end=beta_kwargs['beta_end']) 
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps, beta_s=beta_kwargs['beta_s'])
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # whether to autonormalize

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, condition=None, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x=x, time=t, x_self_cond=x_self_cond, condition=condition)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, condition=None, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x=x, t=t, x_self_cond=x_self_cond, condition=condition)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, condition=None, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition=condition, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None):
        """ given a tuple shape (BxSxE), generate a sample
        """
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        if not (condition is None):
            assert shape[0] == condition.shape[0], f"Condition does not match noise shape. img {shape}, condition {condition.shape}"
            assert self.model.is_conditional, "Trying to condition an unconditional model"
            condition = condition.to(device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(x=img, t=t, condition=condition, x_self_cond=self_cond)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, clip_denoised = True, condition=None, q_sample_idx=0, gt_elem=None, eta_steps=25):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        if q_sample_idx > 0:
            # the sequence has a few elements that are already known, use the GT for them
            assert gt_elem is not None, "Need to provide the GT element for the q_sample_idx"
            if not isinstance(gt_elem, torch.Tensor):
                gt_elem = torch.tensor(gt_elem, device=device)
            if gt_elem.numel() == 1:
                gt_elem = torch.ones(*shape).to(device) * gt_elem

            if gt_elem.shape[0] != batch:
                raise ValueError(f"GT element shape {gt_elem.shape} does not match batch size {batch}")
            
            replacements = iter([self.q_sample(x_start=gt_elem, t=torch.full((batch,), t[0], device=device, dtype=torch.long), noise=None) for t in time_pairs])

        img = torch.randn(shape, device = device)
        x_start = None

        _iter = 0
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            # preds = self.model_predictions(x=x, t=t, x_self_cond=x_self_cond, condition=condition)

            pred_noise, x_start, *_ = self.model_predictions(x=img, t=time_cond, x_self_cond=self_cond, clip_x_start = clip_denoised, condition=condition)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            
            if q_sample_idx > 0:
                # replace the element with the GT
                img[:, :, :q_sample_idx] = next(replacements)[:, :, :q_sample_idx]
            
            if _iter > eta_steps:
                eta = 0.
            _iter += 1

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16, condition=None):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        # TODO: change ddim_sample as well
        if not (condition is None):
            added_seq_len = condition.shape[-1]
        else:
            added_seq_len = 0

        return sample_fn(shape=(batch_size, channels, seq_length-added_seq_len), condition=condition)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, condition=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        
        # set up self denoising
        if self.is_self_denoising and random() < 0.5:
            with torch.no_grad():
                pred_x_start = self.model_predictions(x, t, condition=condition).pred_x_start.detach()
                # print('noise', torch.min(noise), torch.max(noise), noise.shape)
                # print('x', torch.min(x), torch.max(x), x.shape)
                # quit()
                # x = x - noise
                x = self.q_sample(x_start = pred_x_start, t = t, noise = noise)



        # predict and take gradient step

        model_out = self.model(x=x, time=t, x_self_cond=x_self_cond, condition=condition)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        loss_fn = F.mse_loss if self.loss_type == 'l2' else F.l1_loss
        loss = loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()
    
    def p_losses_with_latent(self, x_start, t, noise = None, condition=None):
        b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()
        
        # set up self denoising
        if self.is_self_denoising and random() < 0.5:
            with torch.no_grad():
                pred_x_start = self.model_predictions(x, t, condition=condition).pred_x_start.detach()
                x = self.q_sample(x_start = pred_x_start, t = t, noise = noise)

        # predict and take gradient step
        model_out = self.model(x=x, time=t, x_self_cond=x_self_cond, condition=condition)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = F.l1_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)

        # predict the latent
        pred_x_start = self.predict_start_from_v(x_t=x, t=t, v=model_out)
        return loss.mean(), torch.clamp(pred_x_start, min=-1, max=1), extract(self.loss_weight, t, loss.shape)

    def forward(self, img, *args, **kwargs):
        return img

    # TODO: monkey patch forward based on the instantiation.
    def forward_wo_latents(self, img, *args, **kwargs):
        # print(img.shape)
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        condition = kwargs.get('condition', None)
        len_condition = 0 if (condition is None) else condition.shape[-1]
        # print(len_condition, n, seq_length)
        assert n + len_condition == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return {'diffusion_loss': self.p_losses(img, t, *args, **kwargs)}

    def forward_w_latents(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        condition = kwargs.get('condition', None)
        len_condition = 0 if (condition is None) else condition.shape[-1]
        assert n + len_condition == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses_with_latent(img, t, *args, **kwargs)
