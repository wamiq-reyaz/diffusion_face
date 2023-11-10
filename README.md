## Next3D: Generative Neural Texture Rasterization for 3D-Aware Head Avatars

![Teaser image](./docs/teaser_v3.jpeg)

## Environment
The environment lives in `<root_folder>/env`. In order to add or remove from the environment use the following modification from `<root_folder>`

```bash
mamba <install|uninstall> -p ./env
```
The `-p` stands for prefix (or location). Change the `./env` depending on the relative location.

## Main commands.
A preview of all the available commands is present in `playground/commands.md`. I will keep on updating it as more functionality is added to the project.

The main functionality so far is:
1. Saving the training data in `save_samples_next3d.py`.
2. Generating the RGB from latents in `gen_sampled_next3d.py`.
3. Training a diffusion model in `train_diffusion_eg3d_wplus.py`.

## Faceverse processing

The faceverse environment uses Jittor. Which does JIT compiling - we used CUDA 11.6.2. Download everything with mamba using the `nvidia/proper_cuda_variant/` channel - cuda-nvcc, cudart-dev, cublas-dev, cunpp and a host of other files - check the env file. Then use the GCC version from the `module load command`, we used `11.1.0`. And use `nvcc_path = $(which nvcc)` from an activated conda environment

bfm from `https://github.com/jadewu/3D-Human-Face-Reconstruction-with-3DMM-face-model-from-RGB-image/blob/main/BFM/01_MorphableModel.mat`

## DECA processing

`python demo_reconstruct.py -i  /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d -s /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d/deca_results_unposed --saveObj 1 --saveKpt 1 --saveCoeff 1 --load_eye_pose 1 --eyepath /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d/gaze_results/param --rasterizer_type pytorch3d`

For DECA, you have to replace this line from chumpy `/home/parawr/Projects/Next3D/env/lib/python3.9/site-packages/chumpy/__init__.py`


### TODO:
Currently most commands do not have proper configuration or logging. I am waiting to get to KAUST to setup wandb and SLURM, for a more reproducible pipeline. Until then, everything is hardcoded 🤷‍♂️.

There is a lot of functionality in `notebooks/demo.ipynb`. This includes sampling, guided diffusion. If you need help, I will help you guide through it, but it needs serious refactoring before it is useful to be released.

The env is not yet setup to perform the entire EG3D preprocessing or detection of landmarks for Next3D. This restricts us from using multiple identities. 

**WARNING**
The environment to reproduce the whole preprocessing pipeline seems to be inconsistent. with different parts requiring different version of both PyTorch and Tensorflow. I am currently just setting up a different env for each step in `dataset_preprocessing/ffhq/preprocess_in_the_wild.py`

In order to preprocess an in the wild, run the standard EG3D pipeline and use the gaze estimation and the deep3drecon to get camera and gaze parameters respectively. In addtion, get the mesh using the DECA pipeline.