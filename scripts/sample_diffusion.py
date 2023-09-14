# -----------------------------------------------
# Setup the generator and camera parameters for sampling
# -----------------------------------------------
import os
import sys
sys.path.append('../src')

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import tempfile
from torch.utils.data import DataLoader, DistributedSampler
from itertools import cycle

import legacy as legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training_avatar_texture.triplane_next3d import TriPlaneGenerator
import dnnlib as dnnlib
from gen_samples_next3d import PATTERN, w_plus_hook, set_replacement_hook, WS

from training.dataset import ImageFolderDataset

from training_diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import (
    Unet1D,
    GaussianDiffusion1D
)

import numpy as np
from PIL import Image

import tqdm

def writer(pid, q):
    # poll the queue and write to file
    try:
        while True:
            data = q.get()
            ii = data['ii']
            rank = data['rank']
            img = data['img']
            if img is not None:
                for jj in range(32):
                        Image.fromarray(img[jj]).save(f'/home/parawr/samples_diffusion_10_ddim50_all_poses_cameras/rank_{rank}_{ii*32+jj}.png')
            else:
                return
    finally:
        print(f'Writer {pid} done!')



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
            new_init_kwargs = dict(G.init_kwargs)
            new_init_kwargs['topology_path'] = '../data/ffhq/head_template.obj'
            G_new = TriPlaneGenerator(*G.init_args, **new_init_kwargs).eval().requires_grad_(False).to(device)
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

        # Generate images to start compilation of the cuda ops
            z = torch.from_numpy(np.random.RandomState(seed=0).randn(1, G.z_dim)).to(device)

            angle_p = -0.2
            for angle_y, angle_p in [(0, angle_p)]:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)

                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        
        return G, camera_params, v

# -----------------------------------------------
# Normalization functions
# -----------------------------------------------
def unnormalize_latents(w):
    stats = torch.load('/datawaha/cggroup/parawr/Projects/diffusion/data/gen_samples/zero_pose_50k/stats.pt')
    _min = stats['min'].cuda()
    _max = stats['max'].cuda()
    _range = _max - _min # 73 x 512

    _MIN = torch.nn.functional.pad(_min, (0, 0, 11, 0))
    _RANGE = torch.nn.functional.pad(_range, (0, 0, 11, 0))

    return (w * _RANGE) + _MIN


# -----------------------------------------------
# Generation functions
# -----------------------------------------------
@torch.no_grad()
def generate_images(G, camera_params, v, w, device='cuda', batched=False, **kwargs):
    bs = w.shape[0]
    w = unnormalize_latents(w) # B x 84 x 512
    w = w[:, 11:, :] # B x 73 x 512

    # repeat camera params, and verts
    # camera_params = camera_params.repeat(bs, 1)
    # v = v.repeat(bs, 1, 1)
    # print(w.shape, v.shape, camera_params.shape)

    all_hooks = set_replacement_hook(generator=G,
                                     names=WS,
                                     tensors=w,
                                     batched=batched)
    dummy_w = G.mapping(torch.randn(bs, G.z_dim, device=device),
                        camera_params)

    img = G.synthesis(torch.randn_like(dummy_w, device=w.device), c=camera_params, v=v, noise_mode='const', **kwargs)['image']
    return img

def to_uint8(x):
    return (x.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

def batch_to_PIL(x):
    return [Image.fromarray(t) for t in to_uint8(x).cpu().numpy()]


# -----------------------------------------------
# Create the functions for DDPM sampling 
# -----------------------------------------------
@torch.no_grad()
def p_sample(model, x, t: int, x_self_cond = None, cond_fn=None, guidance_kwargs=None, condition=None):
    """ model is not the unet, but the diffusion model which wraps the unet
    """
    b, *_, device = *x.shape, x.device
    batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
    model_mean, variance, model_log_variance, x_start = model.p_mean_variance(
        x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True,
        condition=condition
    )    
    noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start


@torch.no_grad()
def p_sample_loop(model, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None, condition=None):
    batch, device = shape[0], model.betas.device

    img = torch.randn(shape, device = device)
    imgs = [img]

    x_start = None

    # for t in tqdm(reversed(range(0, model.num_timesteps)), desc = 'sampling loop time step', total = model.num_timesteps):
    for t in reversed(range(0, model.num_timesteps)):
        self_cond = x_start if model.self_condition else None
        img, x_start = p_sample(model, img, t, self_cond, cond_fn, condition=condition, guidance_kwargs=guidance_kwargs)
        imgs.append(img)

    ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

    ret = model.unnormalize(ret)
    return ret


def worker(rank, args):
    # -----------------------------------------------
    # Setup
    # -----------------------------------------------
    init_file = os.path.abspath(os.path.join(args['temp_dir'], '.torch_distributed_init'))
    dist.init_process_group(backend='nccl', init_method=f'file://{init_file}', world_size=args['world_size'], rank=rank)

    torch.cuda.set_device(rank)
    np.random.RandomState(seed=rank)
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)

    # -----------------------------------------------
    # Setup the DataLoaders
    # -----------------------------------------------
    dataset = ImageFolderDataset(path=args['dataset_path'],
                                mesh_path=args['mesh_path'],
                                mesh_type='.obj',
                                load_exp=True,
                                load_lms=True,
                                use_labels=True)
    # Warning: We do not use the set_epoch function, 
    # as we can live with the same ordering across epochs
    sampler = DistributedSampler(dataset, shuffle=False)
    args['dataloader'] = DataLoader(dataset, 
                                    batch_size=args['batch_size'],
                                    shuffle=False,
                                    num_workers=8,
                                    sampler=sampler,
                                    pin_memory=True)

    # -----------------------------------------------
    # Instantiate the Generator
    # -----------------------------------------------
    try:
        del generator
    except Exception as e:
        print(e)
        
    generator, camera_params, verts = setup_generator(network_pkl='../pretrained_models/next3d_ffhq_512.pkl',
                                        obj_path='../data/demo/demo.obj',
                                        lms_path='../data/demo/demo_kpt2d.txt',
                                        fov_deg=18.837,
                                        device='cuda',
                                        reload_modules=False,
                                        )

    for name, module in generator.named_modules():
        module.requires_grad_(False)

    # -----------------------------------------------
    # Create and instantiate the Diffusion Model
    # -----------------------------------------------

    model = Unet1D(
        dim = 512,
        channels=512,
        dim_mults = (1, 2, 4),
        out_dim = 512,
        is_conditional=False,
        add_condition=False
        )

    diffusion = GaussianDiffusion1D(
            model,
            seq_length = 84,
            timesteps = 1000,
            objective = 'pred_v',
            is_self_denoising=False,
            sampling_timesteps=50
        )

    ckpt = torch.load('/datawaha/cggroup/parawr/Projects/diffusion/experiments/unconditoinal_1e-4_32*8_emadecay0.9999_final_final2/model-10.pt',
                    map_location='cpu')

    missing, unk = diffusion.load_state_dict(ckpt['model'])

    print(f'Loaded weights on rank {rank}')

    # -----------------------------------------------
    # Sample from Diffusion and render the images
    # -----------------------------------------------
    
    _iterator = tqdm.trange(391) if rank == 0 else range(391)
    diffusion = diffusion.cuda()
    dloader_iterator = iter(cycle(args['dataloader']))
    for ii in _iterator:
        BATCH_SIZE = args['batch_size']
        new_samples = diffusion.ddim_sample(shape=(BATCH_SIZE, 512, 84))
        # new_samples = p_sample_loop(model=diffusion.cuda(),
        #                             shape=(BATCH_SIZE, 512, 84),
        #                             cond_fn=None
        #                             )
        new_samples = new_samples.permute(0, 2, 1).contiguous() # B x 84 x 512

        # get the next batch of cameras and verts
        _, c, v = next(dloader_iterator)

        # render the images
        img = generate_images(G=generator, 
                                    camera_params=c.cuda(non_blocking=True),
                                    v=v.cuda(non_blocking=True),
                                    w=new_samples,
                                    batched=True)

        # save the images
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img.clone().detach().cpu().numpy()

        # put the images in a queue to be saved by the main process
        args['queue'].put({'img': img,
                    'rank': rank,
                    'ii': ii
                    })

    print('Completed rank', rank)

if __name__ == '__main__':
    mp.set_start_method('spawn')

    # -----------------------------------------------
    # Initalize StyleGan once
    # -----------------------------------------------
    with torch.no_grad():
        generator, camera_params, verts = setup_generator(network_pkl='../pretrained_models/next3d_ffhq_512.pkl',
                                        obj_path='../data/demo/demo.obj',
                                        lms_path='../data/demo/demo_kpt2d.txt',
                                        fov_deg=18.837,
                                        device='cuda',
                                        reload_modules=False,
                                        )

        image_tensor = generate_images(G=generator, 
                               camera_params=camera_params,
                               v=verts,
                               w=torch.randn(1, 84, 512).cuda(),
                               batched=True)
        
        del generator, camera_params, verts, image_tensor
        torch.cuda.empty_cache()
        import gc
        gc.collect()


    # -----------------------------------------------
    # Create a queue to share the results across processes
    # -----------------------------------------------
    queue = mp.Queue()
    args = {'queue': queue}
    print('Queue created')

    # create 10 processes to save the images
    processes = []
    for idx in range(10):
        p = mp.Process(target=writer, args=(idx, args['queue'], ))
        p.start()
        processes.append(p)
    print('Writer processes created')

    # -----------------------------------------------
    # Create the dataset params for the vertices, landmarks, and camera parameters
    # -----------------------------------------------
    args['dataset_path'] = '/datawaha/cggroup/parawr/Projects/diffusion/all_data/ffhq_512_posed_eg3d/'
    args['mesh_path'] = '/datawaha/cggroup/parawr/Projects/diffusion/all_data/ffhq_512_posed_eg3d/deca_results_unposed'
    args['batch_size'] = 32

    # -----------------------------------------------
    # Create the producer processes
    # -----------------------------------------------
    with tempfile.TemporaryDirectory() as temp_dir:
        args['temp_dir'] = temp_dir
        args['world_size'] = torch.cuda.device_count()
        print(args)
        if args['world_size'] == 1:
            print('Running on single GPU')
            worker(rank=0, args=(args, ))
        else:
            print(f'Running on {args["world_size"]} GPUs')
            mp.spawn(fn=worker, args=(args,), nprocs=args['world_size'], join=True)

    print('Producer done')
    print('Sending None to queue')
    for ii in range(100):
        args['queue'].put({'img': None,
                        'rank': 0,
                        'ii': None
                        })

    for p in processes:
        p.join()

    print('Consumers done')
    
    queue.close()
    print('Queue closed')
    
    print('Done')