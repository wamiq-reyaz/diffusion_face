# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import json


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training_avatar_texture.triplane_next3d import TriPlaneGenerator


from functools import partial
ALL_DICT = dict()

def w_plus_hook(name, module, args, output):
    ALL_DICT[name] = output.clone().detach().cpu()

    return torch.ones_like(output)

def replace_hook(name, tensor, module, args, output):
    tensor = tensor.to(output.device)
    valid_len = output.shape[-1]
    return tensor[..., :valid_len] # replace the output with tensor
    

import re
PATTERN = r'.*block[0-9]+$'
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

def set_fwd_hook(generator: torch.nn.Module) -> List:
    all_hooks = []
    for name, module in generator.named_modules():
        if 'affine' in name and 'super' in name:
            mod_hook = partial(w_plus_hook, name)
            hook = module.register_forward_hook(mod_hook)
            all_hooks.append(hook)
    return all_hooks


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

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--obj_path', type=str, help='Path of obj file', required=True)
@click.option('--lms_path', type=str, help='Path of landmark file', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--lms_cond', help='If condition 2d landmarks?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    obj_path: str,
    lms_path: str,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    lms_cond: bool,
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

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
    v = torch.from_numpy(v).cuda().float().unsqueeze(0)

    if lms_cond:
        lms = np.loadtxt(lms_path)
        lms = torch.from_numpy(lms).cuda().float().unsqueeze(0)
        v = torch.cat((v, lms), 1)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed=seed).randn(1, G.z_dim)).to(device)

        angle_p = -0.2
        for idx in range(25):
            imgs = []

            for angle_y, angle_p in [(.4, angle_p), (0, angle_p), (-.4, angle_p)]:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)

                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                
                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

                # print(ws.shape)

                #------------------------- z-norm
                # with open('/storage/nfs/wamiq/next3d/data/generated_samples/attempt1/all_ws.pkl', 'rb') as fd:
                    # import pickle
                    # ws = pickle.load(fd)
                    # print(ws.shape)
                # from stats import mean, std
                # chosen = np.load('notebooks/aa.npy')
                # chosen = torch.from_numpy(chosen).view(1, 512).cuda()
                # # chosen = (ws[42767:42768,0, ...]).cuda()# * torch.from_numpy(std[None, :]).cuda() )+ torch.from_numpy(mean[None, :]).cuda()
                # print(chosen.min(), chosen.max())
                # bb = chosen.unsqueeze(1).repeat_interleave(28, dim=1)
                # print((ws-bb).sum(dim=-1))
                # quit()

                # -------------------- min-max 
                # from stats import  _min, _max
                # _range = _max - _min
                # aa = torch.load('/storage/nfs/wamiq/next3d/min_max/sample-200.png')
                # chosen = (aa[22:23,0, ...]).cuda()* torch.from_numpy(_range[None, :]).cuda() + torch.from_numpy(_min[None, :]).cuda()
                # print(chosen.min(), chosen.max())
                # ws = chosen.unsqueeze(1).repeat_interleave(28, dim=1)


                # print(z.shape)
                # quit()
                # replacement = torch.load('/storage/nfs/wamiq/next3d/min_max_73_conditional_img_noadd_64_no_norm/sample-49.png')
                # # replacement = torch.load('/storage/nfs/wamiq/next3d/min_max_73_segmap/sample-1.png')
                # # replacement = torch.load('/storage/nfs/wamiq/next3d/notebooks/000_simple_train_inversion5.pt')
                # stats = torch.load('./data/generated_samples/w_plus/stats/stats.pt')
                # _min = stats['min'].cuda()
                # _max = stats['max'].cuda()
                # _range = _max - _min
                # # replacement = replacement[idx, :, 6:-4].permute(1,0).cuda()
                # # print(replacement.shape)
                # replacement = replacement[idx, :, 7:].permute(1,0).cuda()
                # # replacement = (replacement + 1) / 2

                # replacement = (replacement * _range) + _min
                # replacement.requires_grad_(True)
                # print(torch.min(replacement[1]), torch.max(replacement[1]))
                # quit()
                # print(replacement.shape, ws.shape)
                # all_hooks = set_replacement_hook(G, WS, replacement)
                img = G.synthesis(ws, camera_params, v, noise_mode='const')['image']
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)

            img = torch.cat(imgs, dim=2)

            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/000_rgb_train_inversion5_{str(idx).zfill(5)}_{seed:04d}.png')

        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, v, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
