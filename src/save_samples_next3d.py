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
import random
import pickle


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training_avatar_texture.triplane_next3d import TriPlaneGenerator
from training_avatar_texture.networks_stylegan2_styleunet import SynthesisBlock as CondSynthesisBlock
from training_avatar_texture.networks_stylegan2 import SynthesisBlock 


from functools import partial
import re

PATTERN = r'.*block[0-9]+$'
ALL_DICT = list()

def w_plus_hook(name, module, args, output):
    ALL_DICT.append((name, output.detach().cpu()))
    return None

def set_fwd_hook(generator: torch.nn.Module) -> List:
    all_hooks = []
    for name, module in generator.named_modules():
        if 'affine' in name:
            mod_hook = partial(w_plus_hook, name)
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
@click.option('--only_frontal', help='Only render from the frontal view, otherwise three side views', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@torch.no_grad()
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
    only_frontal: bool
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
        G._init_kwargs['topology_path'] = '../data/ffhq/head_template.obj'
        print(G.init_kwargs)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G_new.topology_path = os.path.join('..', G.topology_path)
        G = G_new

    returned_wplus_hooks = set_fwd_hook(G)

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

    v = v.repeat_interleave(16, dim=0)


    # Generate images.
    random.seed(seeds[0])
    seeds = []
    for ii in range(100000):
        seeds.append(random.randint(0, 2**32 - 1))


    all_ws = dict()
    with torch.no_grad():
        all_seeds = []
        all_zs = []
        for seed_idx, seed in tqdm(enumerate(seeds), total=len(seeds)):
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
            all_seeds.append(seed)
            all_zs.append(z)
        
        all_zs = torch.cat(all_zs, dim=0)
        z_iterator = torch.split(all_zs, 16, dim=0)

        for curr_idx, z in tqdm(enumerate(z_iterator), total=len(z_iterator)):
            z = z.to(device)
            start = curr_idx * 16
            end = (curr_idx + 1) * 16

            angle_p = -0.2
            if only_frontal:
                angles = [(0, angle_p)]
            else:
                angles = [(.4, angle_p), (0, angle_p), (-.4, angle_p)]

            for angle_y, angle_p in angles:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)

                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                # print(z.shape, conditioning_params.shape)
                # quit()
                bs = z.shape[0]
                conditioning_params = conditioning_params.repeat_interleave(bs, dim=0)
                camera_params = camera_params.repeat_interleave(bs, dim=0)
                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

                # clear the list before synthesising
                global ALL_DICT
                ALL_DICT = []
                all_vals = G.synthesis(ws, camera_params, v, noise_mode='const')
                # print(all_vals['image'].shape)
                # quit()

                # --------------------------------- Start saving ------------------------ #
                img = all_vals['image']
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                ALL_DICT.append(('img', img.clone().detach().cpu()))

                # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/hires_img_{seed:04d}.png')
                
                for ii in range(start, end):
                    all_ws[all_seeds[ii]] = [(k[0], k[1][ii:ii+1]) for k in ALL_DICT]

            if curr_idx % 1000 == 0:
                with open(f'{outdir}/all_wpluss_{curr_idx}.pkl', 'wb') as fd:
                    pickle.dump(all_ws, fd)

                # try removing the previous one to save on space
                try:
                    prev_multiplier = curr_idx // 1000
                    prev_idx = (prev_multiplier - 1) * 1000
                    os.remove(f'{outdir}/all_wpluss_{prev_idx}.pkl')
                except Exception as e:
                    print(e)

        with open(f'{outdir}/all_wpluss.pkl', 'wb') as fd:
            pickle.dump(all_ws, fd)
            



            # # ---------------------- Exploring the mapping of W-affine layers -------- #
            # collected = []
            # import re

            # raw_expr = r'.*conv.*.affine'
            # # expr = re.compile(raw_expr)

            # # print(G)
            # # quit()

            # normal_blocks = []
            # cond_blocks = []
            # for n, m in G.named_modules():
            #     if isinstance(m, SynthesisBlock):
            #         normal_blocks.append((n, m))
            #     if isinstance(m, CondSynthesisBlock):
            #         cond_blocks.append((n, m))
            #     # print(re.findall(raw_expr, n))
            #     if len(re.findall(raw_expr, n)) > 0:
            #         collected.append((n, m))

            # print('*'*10)
            # for k, m in collected:
            #     print(k, super(type(m)))
            # print(len(k))

            # print('*'*10);
            # for k, m in normal_blocks:
            #     print(k)
            # print(len(normal_blocks))

            # print('*'*10);
            # print('cond blocks')
            # for k, m in cond_blocks:
            #     print(k)
            # print(len(normal_blocks))
            # quit()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
