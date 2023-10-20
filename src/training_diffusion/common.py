import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from functools import partial
from typing import Any, List, Dict, Tuple, Union

import torch
import dnnlib
import pickle
from PIL import Image
from camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from torch_utils import misc
import numpy as np

from training_avatar_texture.triplane_next3d import TriPlaneGenerator
import legacy

# ----------------------------------------------------------------------------
WS = ['texture_backbone.synthesis.b4.conv1.affine',
 'texture_backbone.synthesis.b4.torgb.affine',
 'texture_backbone.synthesis.b8.conv0.affine',
 'texture_backbone.synthesis.b8.conv1.affine',
 'texture_backbone.synthesis.b8.torgb.affine',
 'texture_backbone.synthesis.b16.conv0.affine',
 'texture_backbone.synthesis.b16.conv1.affine',
 'texture_backbone.synthesis.b16.torgb.affine',
 'texture_backbone.synthesis.b32.conv0.affine',
 'texture_backbone.synthesis.b32.conv1.affine',
 'texture_backbone.synthesis.b32.torgb.affine',
 'texture_backbone.synthesis.b64.conv0.affine',
 'texture_backbone.synthesis.b64.conv1.affine',
 'texture_backbone.synthesis.b64.torgb.affine',
 'texture_backbone.synthesis.b128.conv0.affine',
 'texture_backbone.synthesis.b128.conv1.affine',
 'texture_backbone.synthesis.b128.torgb.affine',
 'texture_backbone.synthesis.b256.conv0.affine',
 'texture_backbone.synthesis.b256.conv1.affine',
 'texture_backbone.synthesis.b256.torgb.affine',
 'mouth_backbone.synthesis.b8.conv0.affine',
 'mouth_backbone.synthesis.b8.conv1.affine',
 'mouth_backbone.synthesis.b8.torgb.affine',
 'mouth_backbone.synthesis.b16.conv0.affine',
 'mouth_backbone.synthesis.b16.conv1.affine',
 'mouth_backbone.synthesis.b16.torgb.affine',
 'mouth_backbone.synthesis.b32.conv0.affine',
 'mouth_backbone.synthesis.b32.conv1.affine',
 'mouth_backbone.synthesis.b32.torgb.affine',
 'mouth_backbone.synthesis.b64.conv0.affine',
 'mouth_backbone.synthesis.b64.conv1.affine',
 'mouth_backbone.synthesis.b64.torgb.affine',
 'mouth_backbone.synthesis.b128.conv0.affine',
 'mouth_backbone.synthesis.b128.conv1.affine',
 'mouth_backbone.synthesis.b128.torgb.affine',
 'mouth_backbone.synthesis.b256.conv0.affine',
 'mouth_backbone.synthesis.b256.conv1.affine',
 'mouth_backbone.synthesis.b256.torgb.affine',
 'neural_blending.synthesis.b64.conv0.affine',
 'neural_blending.synthesis.b64.conv1.affine',
 'neural_blending.synthesis.b64.torgb.affine',
 'neural_blending.synthesis.b128.conv0.affine',
 'neural_blending.synthesis.b128.conv1.affine',
 'neural_blending.synthesis.b128.torgb.affine',
 'neural_blending.synthesis.b256.conv0.affine',
 'neural_blending.synthesis.b256.conv1.affine',
 'neural_blending.synthesis.b256.torgb.affine',
 'backbone.synthesis.b4.conv1.affine',
 'backbone.synthesis.b4.torgb.affine',
 'backbone.synthesis.b8.conv0.affine',
 'backbone.synthesis.b8.conv1.affine',
 'backbone.synthesis.b8.torgb.affine',
 'backbone.synthesis.b16.conv0.affine',
 'backbone.synthesis.b16.conv1.affine',
 'backbone.synthesis.b16.torgb.affine',
 'backbone.synthesis.b32.conv0.affine',
 'backbone.synthesis.b32.conv1.affine',
 'backbone.synthesis.b32.torgb.affine',
 'backbone.synthesis.b64.conv0.affine',
 'backbone.synthesis.b64.conv1.affine',
 'backbone.synthesis.b64.torgb.affine',
 'backbone.synthesis.b128.conv0.affine',
 'backbone.synthesis.b128.conv1.affine',
 'backbone.synthesis.b128.torgb.affine',
 'backbone.synthesis.b256.conv0.affine',
 'backbone.synthesis.b256.conv1.affine',
 'backbone.synthesis.b256.torgb.affine',
 'superresolution.block0.conv0.affine',
 'superresolution.block0.conv1.affine',
 'superresolution.block0.torgb.affine',
 'superresolution.block1.conv0.affine',
 'superresolution.block1.conv1.affine',
 'superresolution.block1.torgb.affine']
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def replace_hook(name, tensor, module, args, output):
    tensor = tensor.to(output.device)
    valid_len = output.shape[-1]
    return tensor[..., :valid_len] # replace the output with tensor
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def set_replacement_hook(generator: torch.nn.Module, names, tensors, batched=False) -> List:
    all_hooks = []
    for ii, name in enumerate(names):
        for modname, module in generator.named_modules():
            if modname == name:
                if not batched:
                    mod_hook = partial(replace_hook, name, tensors[ii:ii+1, ...])
                else:
                    mod_hook = partial(replace_hook, name, tensors[:, ii, ...])
                hook = module.register_forward_hook(mod_hook)
                all_hooks.append(hook)
    
    return all_hooks
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def create_network_from_pkl(fname, 
                            device: Union[torch.device, str] = "cuda", 
                            reload_modules=False,
                            verbose=False):
    """Create a network from a pkl file.

    Args:
        fname (PathLike): Path to the pkl file.
        device (Union[torch.device, str], optional): Device to load the network to. Defaults to "cuda".
        verbose (bool, optional): Whether to print the network architecture. Defaults to False.

    Returns:
        torch.nn.Module: The network.
    """
    with dnnlib.util.open_url(fname) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        if verbose:
            print(G)
    if reload_modules:
        print('Reloading modules...')
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    return G
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class Unnormalizer:
    def __init__(self, path, mode='min_max'):
        self.mode = mode
        stats = torch.load(path, map_location='cpu')
        if mode == 'min_max':
            self.min = torch.from_numpy(stats['min'].numpy())
            try:
                self.range = torch.from_numpy(stats['range'].numpy())
            except KeyError:
                self.max = torch.from_numpy(stats['max'].numpy())
                self.range = self.max - self.min
            self.range[self.range == 0] = 1.
        elif mode == 'z_norm':
            self.mean = torch.from_numpy(stats['mean'].numpy())
            self.std = torch.from_numpy(stats['std'].numpy())
            self.std[torch.isnan(self.std)] = 1.

    def __call__(self, w):
        device = w.device
        if self.mode == 'min_max':
            return w * self.range[None, ...].to(device) + self.min[None, ...].to(device)
        elif self.mode == 'z_norm':
            return w * self.std[None, ...].to(device) + self.mean[None, ...].to(device)
        else:
            raise NotImplementedError
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_images(G, camera_params, v, w,
                    device='cuda', 
                    batched=False,
                    unnormalizer=None,
                     **kwargs):
    bs = w.shape[0]
    w = unnormalizer(w) # B x 73 x 512

    all_hooks = set_replacement_hook(generator=G,
                                     names=WS,
                                     tensors=w,
                                     batched=batched)
    dummy_w = G.mapping(torch.randn(bs, G.z_dim, device=device),
                        camera_params)

    img = G.synthesis(torch.randn_like(dummy_w, device=w.device), c=camera_params, v=v, noise_mode='const', **kwargs)['image']
    return img
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def get_vertices(obj_path, lms_path, lms_cond, device='cpu'):
    # load fixed vertices
    v = []
    with open(obj_path, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            if line[:2] == "v ":
                v.append([float(x) for x in line.split()[1:]])
    v = np.array(v).reshape((-1, 3))
    v = torch.from_numpy(v).float().unsqueeze(0)

    if lms_cond:
        lms = np.loadtxt(lms_path)
        lms = torch.from_numpy(lms).float().unsqueeze(0)
        v = torch.cat((v, lms), 1)

    return v.to(device)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def get_lookat_cam(angle_y=0.0, angle_p=-0.2, cam_pivot=[0, 0, 0.2], cam_radius=2.7, fov_deg=18.837, device='cpu'):
    cam_pivot = torch.tensor(cam_pivot, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    return camera_params
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def to_uint8(x):
    return (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def batch_to_PIL(x):
    np_images = to_uint8(x).cpu().numpy()
    pil_images = []
    for img_np in np_images:
        if img_np.shape[-1] == 1:  # Grayscale
            pil_images.append(Image.fromarray(img_np[:,:,0], 'L'))
        else:
            pil_images.append(Image.fromarray(img_np))
    return pil_images
# ----------------------------------------------------------------------------
