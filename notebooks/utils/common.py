import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math

import torch
import torch.nn as nn
import tqdm
import numpy as np
from PIL import Image
import re
from hydra import compose, initialize
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# ----------------------------------------------
from src.training_diffusion.models.builder import get_model
from src.training_diffusion.diffusion.builder import get_diffusion
from src.training_diffusion.conditioners.builder import get_conditioner

import src.legacy as legacy
from src.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from src.torch_utils import misc
from src.training_avatar_texture.triplane_next3d import TriPlaneGenerator
import src.dnnlib as dnnlib


# ----------------------------------------------
def update_fn(args, update_args, ckpt_type='latest'):
    ''' Update the args containing the base experiment with the 
    experiment details in update_args
    '''
    s = update_args.expt_name
    match = re.split(r'(_\d{4}-\d{2}-\d{2})', s, maxsplit=1)
    args.base_path = os.path.join(args.base_dir, match[0])
    args.model_path = os.path.join(args.ckpt_path, ''.join(match) + f'_{ckpt_type}.pt')
    args.orig_cwd = update_args.orig_cwd
    return args

# ----------------------------------------------
def register_resolver():
    ''' Registers the hydra resolver needed for computing the actual sequence 
    length
    '''
    from src.infra.utils import difusion_length_resolver
    try:
        OmegaConf.register_new_resolver(
            name='diffusion_length_resolver',
            resolver=difusion_length_resolver,
        )
    except Exception as e:
        print('Resolver already registered')
        print(e)
        pass
    
# ----------------------------------------------
def get_models(cfg):
    ''' Returns the model, conditioner and diffusion objects
    '''
    unet = get_model(cfg)
    diffusion = get_diffusion(cfg, unet)
    conditioner = get_conditioner(cfg)
    
    return conditioner, diffusion

# ----------------------------------------------
def load_weights(args, diffusion, conditioner, ema=True):
    ''' Loads the weights for the diffusion and conditioner models
    '''
    diffusion = diffusion.cuda()
    conditioner = conditioner.cuda()
    
    ckpt = torch.load(os.path.join(args.base_path, args.model_path), map_location='cpu')
    if ema:
        new_weights = dict()
        prefix = 'ema_model.'
        for k,v in ckpt['ema'].items():
            if k.startswith(prefix):
                new_weights[k[len(prefix):]] = v
    else:
        new_weights = ckpt['model']
        
    print('loading diffusion weights')
    missing, unexpected = diffusion.load_state_dict(new_weights, strict=False)
    print('missing keys: ', missing)
    print('unexpected keys: ', unexpected)
    
    print('loading conditioner weights')
    missing, unexpected = conditioner.load_state_dict(new_weights, strict=False)
    print('missing keys: ', missing)
    print('unexpected keys: ', unexpected)
    
    return diffusion, conditioner

# ----------------------------------------------
def setup_generator(network_pkl,
                    obj_path='../data/demo/demo.obj',
                    lms_path='data/demo/demo_kpt2d.txt',
                    fov_deg=18.837,
                    device='cuda',
                    reload_modules=False,
                    ):
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

        if True:
            lms = np.loadtxt(lms_path)
            lms = torch.from_numpy(lms).cuda().float().unsqueeze(0)
            v = torch.cat((v, lms), 1)

        # Generate images.
            z = torch.from_numpy(np.random.RandomState(seed=0).randn(1, G.z_dim)).to(device)

            angle_p = 0.0
            for angle_y, angle_p in [(0, angle_p)]:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)

                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                
        for n, m in G.named_modules():
            m.requires_grad_(False)
        
        return G, camera_params, v
    
# ----------------------------------------------
def get_frontal_trajectory(
    base_pitch=0.2,
    base_yaw=0.4,
    base_fov=18.837,
    base_radius=2.7,
    change_radius=False,
    fov_amplitude=3.0,
    radius_amplitude=0.1,
    num_frames=100,
):
    ''' Returns a frontal trajectory for the camera as a list of tuples
    (pitch, yaw, fov, radius)
    '''
    trajectory = []
    for t in np.linspace(0, 1, num_frames):
        pitch = base_pitch * np.cos(t * 2 * math.pi) + math.pi/2
        yaw = base_yaw * np.sin(t * 2 * math.pi) + math.pi/2
        fov = base_fov

        fov = fov + fov_amplitude + np.sin(t * 2 * math.pi) * fov_amplitude
        
        radius = base_radius
        if change_radius:
            radius = radius + radius_amplitude + np.sin(t * 2 * math.pi) * radius_amplitude

        trajectory.append((pitch, yaw, fov, radius))

    return trajectory

# ----------------------------------------------
def trajectory_to_cams(
    trajectory,
    device='cuda',
):
    ''' from a trajectory of (pitch, yaw, fov, radius) returns a list of camera parameters
    in the eg3d format (bs, 25)
    '''
    camera_params = []
    for pitch, yaw, fov, radius in trajectory:
        cam_pivot = torch.tensor([0, 0, 0.2], device=device)
        cam2world_pose = LookAtPoseSampler.sample(pitch, yaw, cam_pivot, radius=radius, device=device)
        intrinsics = FOV_to_intrinsics(fov, device=device)
        camera_params.append(torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1))
        
    return torch.cat(camera_params, 0)

# ----------------------------------------------
def save_image_grid(images, output_filename, gridsize=10):
    '''
    Saves a list of 100 images as a 10x10 grid.
    Images are plotted without axis or spacing.
    Column and row numbers are added on the topmost and leftmost parts of the images.

    Parameters:
        images (list): List of 100 image file paths.
        output_filename (str): Path for saving the final grid image.
    '''
    if len(images) != (gridsize ** 2):
        raise ValueError("The images list should contain exactly 100 image paths.")
    
    fig, axarr = plt.subplots(gridsize, gridsize, figsize=(2*gridsize, 2*gridsize), dpi=160)

    # Removing spacing between images
    fig.subplots_adjust(hspace=0, wspace=0)

    for i, ax in enumerate(axarr.ravel()):
        # Read the image
        img = images[i]
        ax.imshow(img)

        # Remove axes
        ax.axis('off')

        # Add column numbers (topmost part)
        if i < gridsize:
            ax.text(0.5, 0, str(i), ha='center', va='top', transform=ax.transAxes, color='white')

        # Add row numbers (leftmost part)
        if i % gridsize == 0:
            ax.text(0, 0.5, str(i // gridsize), ha='left', va='center', transform=ax.transAxes, color='white')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
