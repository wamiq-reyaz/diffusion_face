import os
import sys
import pickle
from typing import Any
from training_diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import Trainer1D, GaussianDiffusion1D, Unet1D, TrainerPhotometric

import torch
import torch.nn as nn
import tensorboard as tb
from torch.utils.data import Dataset
import numpy as np
from torchvision.models import resnet18
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

class WData(Dataset):
    def __init__(self,
                path,
                img_path=None,
                normalize=True):
        # TODO configurable image mean and std
        self.path = path
        self.normalize = normalize
        self.img_path  = img_path

        self.data = torch.load(self.path).numpy()[:75000]  # it is a 10000x73x512 w
        stats = torch.load('./data/generated_samples/w_plus/stats/stats.pt')
        print('Loaded data')

        # if self.img_path:
        #     self.img_data = np.load(self.img_path, mmap_mode='r') # tensor of shape Nx3xHxW
        #     self.img_data = self.img_data.astype(np.float32) / 255.0
        #     self.img_data = self.img_data - IMAGENET_MEAN[None, :, None, None]
        #     self.img_data = self.img_data / IMAGENET_STD[None, :, None, None]
        # print('Loaded Image Data')

        # assert self.img_data.shape[0] == self.data.shape[0], "The latent data and the image data sizes do not match"

        if self.normalize:
            _min, _max = stats['min'].numpy(), stats['max'].numpy()
            _range = _max - _min
            _range[_range == 0] = 1.
        
        # self.data = self.data - mean[np.newaxis, :]
        # self.data = self.data / std[np.newaxis, :]

        self.data = self.data - _min[np.newaxis, :]
        self.data = self.data / _range[np.newaxis, :]
        # self.data = (self.data - 0.5) * 2 # range [-1, 1]
        
        self.data = torch.from_numpy(self.data)
        if self.img_path:
            # self.data = nn.functional.pad(self.data, pad=(0, 0, 6, 4)) # pad from last-dim to front. We are trying to make 73 even
            # add 64 resnet feats
            self.data = nn.functional.pad(self.data, pad=(0, 0, 7, 0)) 
        else:
            self.data = nn.functional.pad(self.data, pad=(0, 0, 6, 5)) # pad from last-dim to front. We are trying to make 73 even
        # self.data = nn.functional.pad(self.data, pad=(0, 0, 8, 7)) # pad from last-dim to front. We are trying to make 73 even

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index: Any) -> Any:
        data = self.data[index, :].permute(1,0) # 73x512

        img = Image.open(
                        os.path.join(
                                    self.img_path,
                                    str(index).zfill(7) + '.png')
                                    ).resize(
                                            (256,256,),
                                            resample=Image.Resampling.LANCZOS
                        )
        img = np.array(img).astype(np.float32) / 255.0 # HxWx3
        img = torch.from_numpy(img).permute(2,0,1) # 3xHxW

        img = img - IMAGENET_MEAN[:, None, None]
        img = img / IMAGENET_STD[:, None, None]

        if self.img_path:
            retval = {'data': data,
                       'condition': img} # 3xHxW prenormalized
        else:
            retval = data
        return retval
    

if __name__ == '__main__':
    model = Unet1D(
    dim = 512,
    channels=512,
    dim_mults = (1, 2, 4),
    out_dim = 512,
    is_conditional=True,
    add_condition=False
    )

    # print(model)
    def pytorch_count_params(model):
        from functools import reduce
        "count number trainable parameters in a pytorch model"
        total_params = sum(reduce( lambda a, b: a*b, x.size()) for x in model.parameters())
        return total_params

    print(pytorch_count_params(model=model))

    # conditional_model = resnet18(pretrained=True)
    # conditional_model.fc = nn.Identity()
    # conditional_model.avgpool = nn.Identity()

    class CondModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
            self.model.classifier[4] = torch.nn.Identity()
            self.model.aux_classifier = torch.nn.Identity()
            self.projector = nn.Conv2d(256, 512, 1, 1)

        def forward(self, x):
            # with torch.no_grad():
            x = self.model.backbone(x)
            x = self.model.classifier[0](x['out']) # ASPP
            x = self.model.classifier[1](x) # Conv
            x = self.model.classifier[2](x) # BN
            x = self.model.classifier[3](x) # ReLU

            return self.projector(x)

        # # @ override to yield only projector parameters
        # def parameters(self):
        #     # for p in self.projector.parameters():
        #     #     yield p

        #     for p in self.parameters():
        #         yield p

    conditional_model = CondModel()


    diffusion = GaussianDiffusion1D(
        model,
        seq_length = 336,
        timesteps = 1000,
        objective = 'pred_v',
        is_self_denoising=False
    )

    dataset = WData(path='/storage/nfs/wamiq/next3d/data/generated_samples/w_plus/all_data.pt',
                    img_path='/storage/nfs/wamiq/next3d/data/generated_samples/w_plus_img/images'
    )

    # ------------------- Rendering Params ----------------------------- #
    # imagenet normalization. Should be of shape 1xExS
    stats = torch.load('./data/generated_samples/w_plus/stats/stats.pt')
    _min, _max = stats['min'], stats['max']
    _range = _max - _min
    _range[_range == 0] = 1.

    _range = torch.nn.functional.pad(_range, (0, 0, 7, 0)) # 80x512
    _min = torch.nn.functional.pad(_min, (0, 0, 7, 0)) # 80x512

    _range = _range.permute(1,0).unsqueeze(0) # 1x512x80
    _min = _min.permute(1,0).unsqueeze(0) # 1x512x80

    # eg3d model and camera/vertex params
    import legacy
    from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
    from torch_utils import misc
    from training_avatar_texture.triplane_next3d import TriPlaneGenerator
    import dnnlib

    def setup_generator(network_pkl='./pretrained_models/next3d_ffhq_512.pkl',
                        obj_path='./data/demo/demo.obj',
                        lms_path='./data/demo/demo_kpt2d.txt',
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

                angle_p = -0.2
                for angle_y, angle_p in [(0, angle_p)]:
                    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                    
                    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)

                    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            
            return G, camera_params, v
    
    # generator_model, camera_params, v = setup_generator()

    # for name, module in generator_model.named_modules():
    #     module.requires_grad_(False)


    # ------------------- Rendering Params ----------------------------- #
    # Or using trainer

    trainer = Trainer1D(
        diffusion,
        dataset = dataset,
        train_batch_size = 32*8,
        train_lr = 1e-4,
        train_num_steps = 100000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.9999,                # exponential moving average decay
        ema_update_every=10,
        amp = False,                       # turn on mixed precision
        results_folder='min_max_73_images_deeplab_mobilenet_ft1e-4_yesnorm_actuallyrgb_1e-4_tanh_all_scalecond',
        save_and_sample_every=1000,
        conditional_model=conditional_model,
        normalize_condition=True,
        conditional_lr=1e-4
    )
    # trainer.load_only_weights('/storage/nfs/wamiq/next3d/min_max_73_conditional_img_noadd_64_no_norm/model-49.pt', full_path=True)
    # trainer.load('/storage/nfs/wamiq/next3d/min_max_73_images_resnet_ft1e-4_yesnorm4/model-4.pt', full_path=True)
    trainer.train()

    # trainer = TrainerPhotometric(
    #     diffusion,
    #     dataset = dataset,
    #     train_batch_size = 6*4,
    #     train_lr = 1e-5,
    #     train_num_steps = 100000,         # total training steps
    #     gradient_accumulate_every = 1,    # gradient accumulation steps
    #     ema_decay = 0.9999,                # exponential moving average decay
    #     ema_update_every=10,
    #     amp = False,                       # turn on mixed precision
    #     results_folder='min_max_73_conditional_img_noadd_64_no_norm_35k_photometric_onecyle_1e-5_pohotweight2_weightedphoto_4gpu_redo',
    #     save_and_sample_every=1000,
    #     conditional_model=conditional_model,
    #     normalize_condition=False,
    #     generator_model=generator_model,
    #     dataset_mean = IMAGENET_MEAN[:, None, None],
    #     dataset_std = IMAGENET_STD[:, None, None],
    #     _range = _range, 
    #     _min = _min, 
    #     camera_params = camera_params,
    #     v = v,
    #     photometric_weight=2
    # )
    # trainer.load_only_weights('/storage/nfs/wamiq/next3d/min_max_73_conditional_img_noadd_64_no_norm/model-49.pt', full_path=True)
    # trainer.train()

    # after a lot of training

    sampled_seq = diffusion.sample(batch_size = 4)
    with open('aa.pkl', 'wb') as fd:
        pickle.dump(sampled_seq, fd)
    sampled_seq.shape # (4, 32, 128)
    


