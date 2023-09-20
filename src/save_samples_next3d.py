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
import sys
sys.path.append('../scripts')
import re
from typing import List, Optional, Tuple, Union

import torch
import torch.multiprocessing as mp
if __name__ == '__main__':
    print(mp.set_start_method('spawn'))
import torch.distributed as dist
import click
import dnnlib
import numpy as np
import PIL.Image
from tqdm import tqdm
import mrcfile
import json
import random
import pickle
from joblib import dump
from PIL import Image
import zarr
from numcodecs import Blosc
import tempfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training_avatar_texture.triplane_next3d import TriPlaneGenerator
from training_avatar_texture.networks_stylegan2_styleunet import SynthesisBlock as CondSynthesisBlock
from training_avatar_texture.networks_stylegan2 import SynthesisBlock 
from training.dataset import ImageFolderDataset
from gen_samples_next3d import WS


from torch.utils.data import DataLoader, DistributedSampler
from itertools import cycle

from functools import partial
import re

from sample_diffusion import setup_generator

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

def writer(pid, q):
    try:
        while True:
            data = q.get()
            idx = data['idx']
            img = data['img']
            _dir = data['dir']
            if img is not None:
                for ii, n in enumerate(idx):
                        Image.fromarray(
                                    img[ii], 'RGB').save(
                                        os.path.join(_dir, f'{str(n).zfill(7)}.png'
                                        )
                                    )   
            else:         
                return
    finally:
        print(f'Writer {pid} done!')

def proxy(rank, args):
    kwargs = args
    generate_images(rank, **kwargs)

def generate_images(
    rank: int,
    queue: mp.Queue,
    _array: zarr.Array,
    temp_dir:str,
    sample_cams: bool,
    sample_ids:bool,
    dataset_path: str,
    mesh_path: str,
    batch_size: int,
    world_size: int,
    network_pkl: str,
    seeds: List[int], # These seeds are a fair bit different than the seeds in the main command
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
    only_frontal: bool,
    scale_lms:bool,
    num_samples: int,
    save_images: bool,
    **kwargs
):
    # ----------------------------------------------------------------------------
    # Initialize torch.distributed
    # ----------------------------------------------------------------------------
    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    dist.init_process_group(backend='nccl', init_method=f'file://{init_file}', world_size=world_size, rank=rank)

    # ----------------------------------------------------------------------------
    # Setup
    # ----------------------------------------------------------------------------
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)

    # ----------------------------------------------------------------------------
    # Setup the hooks
    # ----------------------------------------------------------------------------
    PATTERN = r'.*block[0-9]+$'
    ALL_DICT = list()

    def w_plus_hook(name, module, args, output):
        ALL_DICT.append((name, output.clone().detach().cpu()))
        return None

    def set_fwd_hook(generator: torch.nn.Module) -> List:
        all_hooks = []
        for name, module in generator.named_modules():
            if 'affine' in name:
                mod_hook = partial(w_plus_hook, name)
                hook = module.register_forward_hook(mod_hook)
                all_hooks.append(hook)

        return all_hooks

    # ----------------------------------------------------------------------------
    # Load networks 
    # ----------------------------------------------------------------------------
    G, conditioning_params, v = setup_generator(
        network_pkl=network_pkl,
        obj_path=obj_path,
        lms_path=lms_path,
        fov_deg=18.837,
        device='cuda',
        reload_modules=reload_modules,
    )
    conditioning_params_orig = conditioning_params.clone().cuda()
    v_orig = v.clone().cuda()

    for name, module in G.named_modules():
        module.requires_grad_(False)

    returned_wplus_hooks = set_fwd_hook(G)

    print('Networks loaded and hooked')

    # ----------------------------------------------------------------------------
    # Create a dataset that iterates over different camera poses and vertices
    # ----------------------------------------------------------------------------
    if sample_cams or sample_ids:
        dataset = ImageFolderDataset(path=dataset_path,
                                    mesh_path=mesh_path,
                                    mesh_type='.obj',
                                    load_exp=True,
                                    load_lms=True,
                                    use_labels=True)
        # Warning: We do not use the set_epoch function, 
        # as we can live with the same ordering across epochs
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, 
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    sampler=sampler,
                                    pin_memory=True)
    else:
        dataloader = (None, None, None)

    print('Dataset created')
    
    # ----------------------------------------------------------------------------
    # Slice the current portion of seeds
    # ----------------------------------------------------------------------------
    assert num_samples % world_size == 0
    local_len = num_samples // world_size
    seeds = seeds[rank*local_len:local_len*(rank+1)]
    seed_idxes = np.arange(num_samples)[rank*local_len:local_len*(rank+1)]

    # ----------------------------------------------------------------------------
    # Run the main loop
    # ----------------------------------------------------------------------------
    dloader_iterator = iter(cycle(dataloader))
    all_ws = dict()
    with torch.no_grad():
        all_seeds = []
        all_zs = []
        _iterator = tqdm(enumerate(seeds), total=len(seeds)) if rank == 0 else enumerate(seeds)
        for seed_idx, seed in _iterator:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim))
            all_seeds.append(seed)
            all_zs.append(z)
        
        all_zs = torch.cat(all_zs, dim=0).cuda()
        z_iterator = torch.split(all_zs, batch_size, dim=0)

        _iterator = tqdm(enumerate(z_iterator), total=len(z_iterator)) if rank == 0 else enumerate(z_iterator)
        for curr_idx, z in _iterator:
            z = z.cuda()
            bs = z.shape[0]
            start = curr_idx * bs
            end = (curr_idx + 1) * bs
            curr_seed_idxes = seed_idxes[start:end]

            angle_p = -0.2
            if only_frontal:
                angles = [(0, angle_p)]
            else:
                angles = [(.4, angle_p), (0, angle_p), (-.4, angle_p)]

            for angle_y, angle_p in angles:
                # TODO: sample conditioning and camera parameters independently
                if sample_cams or sample_ids:
                    _, conditioning_params, v = next(dloader_iterator)
                    _c = conditioning_params.cuda()
                    _v = v.cuda()
                if not sample_cams:
                    _c = conditioning_params_orig.repeat_interleave(bs, dim=0)
                if not sample_ids:
                    _v = v_orig.repeat_interleave(bs, dim=0)

                ws = G.mapping(z, _c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                # clear the list before synthesising
                ALL_DICT = []
                img = G.synthesis(ws, _c, _v, noise_mode='const')['image']

                # ----------------------------------------------------------------------------
                # Save the images
                # ----------------------------------------------------------------------------
                if save_images:
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    img = img.clone().detach().cpu().numpy()
                    queue.put({'idx': curr_seed_idxes,
                               'img': img,
                               'dir': os.path.join(outdir, 'images')
                               })
                
                # ----------------------------------------------------------------------------
                # Save the arrays
                # ----------------------------------------------------------------------------
                for k in ALL_DICT:
                    valid_len = k[1].shape[-1]
                    data = np.ones((len(curr_seed_idxes), 512), dtype=np.float32)
                    data[:, :valid_len] = k[1].numpy()
                    _array[k[0]]['data'][curr_seed_idxes, :] = data

            

    print('Completed rank', rank)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    @click.command()
    @click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
    @click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
    @click.option('--obj_path', type=str, help='Path of obj file', required=True)
    @click.option('--lms_path', type=str, help='Path of landmark file', required=True)
    @click.option('--dataset_path', type=str, help='Path of Image/Camera dataset', required=True)
    @click.option('--mesh_path', type=str, help='Path of Mesh dataset', required=True)
    @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
    @click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
    @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
    @click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
    @click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
    @click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
    @click.option('--lms_cond', help='If condition 2d landmarks?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--save_images', help='Whether to save the rendered images as well', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
    @click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--scale_lms', help='If 2d landmarks are from DECA', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--only_frontal', help='Only render from the frontal view, otherwise three side views', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--num_samples', help='Number of samples to generate', type=int, required=True, metavar='int', default=100000, show_default=True)
    @click.option('--num_gpus', help='Number of GPUs', type=int, required=False, metavar='int', default=1, show_default=True)
    @click.option('--num_writers', help='Number of concurrent writers', type=int, required=False, metavar='int', default=10, show_default=True)
    @click.option('--sample_cams', help='Sample cameras from the dataset', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--sample_ids', help='Sample Ids from the dataset', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    @click.option('--batch_size', help='Batch size', type=int, required=False, metavar='int', default=32, show_default=True)
    @click.option('--lmdb', help='Whether to use the lmdb backedn', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
    def main(
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
    only_frontal: bool,
    scale_lms:bool,
    num_samples: int,
    num_gpus: int,
    num_writers: int,
    sample_cams: bool,
    sample_ids: bool,
    dataset_path: str,
    mesh_path: str,
    save_images: bool,
    batch_size: int,
    lmdb: bool,
):
        """Generate images using pretrained network pickle.

        Examples:

        \b
        # Generate an image using pre-trained FFHQ model.
        python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
            --network=ffhq-rebalanced-128.pkl
        """
        args = locals()

        # ----------------------------------------------------------------------------
        # Setup
        # ----------------------------------------------------------------------------
        random.seed(seeds[0])
        seeds = []
        for ii in range(num_samples):
            seeds.append(random.randint(0, 2**32 - 1))
        args['seeds'] = seeds

        # ----------------------------------------------------------------------------
        # Create the output directory
        # ----------------------------------------------------------------------------
        os.makedirs(outdir, exist_ok=True)
        if save_images:
            os.makedirs(os.path.join(outdir, 'images'), exist_ok=True)

        # ----------------------------------------------------------------------------
        # Run StyleGAN once to make sure the modules are loaded
        # ----------------------------------------------------------------------------
        G, c, v = setup_generator(
            network_pkl=network_pkl,
            obj_path=obj_path,
            lms_path=lms_path,
            fov_deg=18.837,
            device='cuda',
            reload_modules=False,
        )
        with torch.no_grad():
            w = G.mapping(torch.randn(1, G.z_dim, device=c.device),
                                c)

            _ = G.synthesis(torch.randn_like(w, device=w.device), c=c, v=v, noise_mode='const')['image']

            del G, c, v, w
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()



        # ----------------------------------------------------------------------------
        # Create the zarr array
        # ----------------------------------------------------------------------------
        synchronizer = zarr.ProcessSynchronizer(os.path.join(outdir, 'samples.lock'))
        _store_path = 'samples.lmdb' if lmdb else 'samples.zarr'
        store = zarr.LMDBStore(os.path.join(outdir, _store_path)) if lmdb else zarr.DirectoryStore(os.path.join(outdir, _store_path))
        # store = zarr.DirectoryStore(os.path.join(outdir, _store_path))
        # TODO: make sure we have prompting available
        main_group = zarr.group(store=store, overwrite=True)
        subgroups = []
        for w in WS:
            subgroups.append(main_group.create_group(w))

        for sg in subgroups:
            z = sg.ones(name='data', 
                        compressor=Blosc(cname='zlib', clevel=0, shuffle=Blosc.SHUFFLE),
                        shape=(num_samples, 512),
                        chunks=(1024, 512),
                        dtype='f4',
                        overwrite=True,
                        synchronizer=synchronizer,
                        write_empty_chunks=True)
            # # fill in the chunks upfront
            # z[:] = 1.0
            
        args['_array'] = main_group
        print(main_group[WS[0]]['data'].info)
        print('Zarr array created')
            
        # ----------------------------------------------------------------------------
        # Create the queue
        # ----------------------------------------------------------------------------
        queue = mp.Queue()
        args['queue'] = queue

        # ----------------------------------------------------------------------------
        # Create the writers
        # ----------------------------------------------------------------------------
        writers = []
        for ii in range(num_writers):
            writers.append(mp.Process(target=writer, args=(ii, queue)))
            writers[-1].start()
        print('Writers started')

        # ----------------------------------------------------------------------------
        # Create the producder processes
        # ----------------------------------------------------------------------------
        with tempfile.TemporaryDirectory() as temp_dir:
            args['temp_dir'] = temp_dir
            args['world_size'] = torch.cuda.device_count()
            for k in args:
                if k != 'seeds':
                    print(k, args[k])
            if args['world_size'] == 1:
                print('Running on single GPU')
                proxy(rank=0, args=args)
            else:
                print(f'Running on {args["world_size"]} GPUs')
                mp.spawn(fn=proxy, args=(args,), nprocs=args['world_size'], join=True)

        # ----------------------------------------------------------------------------
        # Kill the writers
        # ----------------------------------------------------------------------------
        print('Producer done, killing writers')
        for ii in range(100*num_writers):
            queue.put({'idx': None, 'img': None, 'dir': None})
        for w in writers:
            w.join()
        print('Writers done')

        queue.close()
        print('Queue closed')

        # ----------------------------------------------------------------------------
        # Save the arguments
        # ----------------------------------------------------------------------------
        save_args = {k:v for k,v in args.items() if k != 'seeds'}
        for k, v in save_args.items():
            try:
                v = str(v)
                save_args[k] = v
            except:
                pass
        with open(os.path.join(outdir, 'args.json'), 'w') as f:
            json.dump(save_args, f)

        # ----------------------------------------------------------------------------
        # Save the seeds
        # ----------------------------------------------------------------------------
        with open(os.path.join(outdir, 'seeds.pkl'), 'wb') as f:
            pickle.dump(seeds, f)

        # ----------------------------------------------------------------------------
        # Save the zarr array
        # ----------------------------------------------------------------------------
        main_group.attrs['seeds'] = seeds
        main_group.attrs['num_samples'] = num_samples
        main_group.attrs['num_writers'] = num_writers
        main_group.attrs['num_gpus'] = num_gpus

        # ----------------------------------------------------------------------------
        # Flush to disk
        # ----------------------------------------------------------------------------
        if lmdb:
            store.flush()
            store.close()
        print('Store flushed to disk')

        # ----------------------------------------------------------------------------
        # End
        # ----------------------------------------------------------------------------
        print('Done')

    # ----------------------------------------------------------------------------

    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
