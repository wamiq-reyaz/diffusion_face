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

### TODO:
Currently most commands do not have proper configuration or logging. I am waiting to get to KAUST to setup wandb and SLURM, for a more reproducible pipeline. Until then, everything is hardcoded 🤷‍♂️.

There is a lot of functionality in `notebooks/demo.ipynb`. This includes sampling, guided diffusion. If you need help, I will help you guide through it, but it needs serious refactoring before it is useful to be released.

The env is not yet setup to perform the entire EG3D preprocessing or detection of landmarks for Next3D. This restricts us from using multiple identities. 

**WARNING**
The environment to reproduce the whole preprocessing pipeline seems to be inconsistent. with different parts requiring different version of both PyTorch and Tensorflow. I am currently just setting up a different env for each step in `dataset_preprocessing/ffhq/preprocess_in_the_wild.py`
