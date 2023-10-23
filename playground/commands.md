### Generation RGB
Path to obama
/storage/nfs/wamiq/next3d/data/reenactment/obama/01491.obj
/storage/nfs/wamiq/next3d/data/reenactment/obama/01491_kpt2d.txt

```bash
python gen_samples_next3d.py --trunc=0.7 --shapes=true --seeds=169 --network=../pretrained_models/next3d_ffhq_512.pkl --obj_path=data/demo/demo.obj --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True --outdir=73_samples --shapes=False

python gen_samples_next3d.py --trunc=0.7 --shapes=False --seeds=169 --network=../pretrained_models/next3d_ffhq_512.pkl --obj_path=/storage/nfs/wamiq/next3d/data/reenactment/obama/01491.obj --lms_path=/storage/nfs/wamiq/next3d/data/reenactment/obama/01491_kpt2d.txt --lms_cond=True --outdir=73_samples --shapes=False

python gen_samples_next3d.py --trunc=0.7 --shapes=False --seeds=169 --network=../pretrained_models/next3d_ffhq_512.pkl --obj_path=/mnt/ibex/Projects/Next3D/data/demo/demo.obj --lms_path=/mnt/ibex/Projects/Next3D/data/demo/demo_kpt2d.txt --lms_cond=True --outdir=/mnt/ibex_ai/Projects/diffusion/scratch/samples_demo --shapes=False


```

### Generation samples

```bash
python save_samples_next3d.py --outdir=out --trunc=0.7 --shapes=False --seeds=169     --network=../pretrained_models/next3d_ffhq_512.pkl --obj_path=data/demo/demo.obj     --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True --only_frontal=True --reload_modules=True --outdir=./data/generated_samples/w_plus_img

python save_samples_next3d.py --outdir=out --trunc=0.7 --shapes=False --seeds=169     --network=../pretrained_models/next3d_ffhq_512.pkl --obj_path=data/demo/demo.obj     --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True --only_frontal=True --reload_modules=True --outdir=/mnt/ibex_ai/Projects/diffusion/data/w_plus_img

python save_samples_next3d.py --outdir=out --trunc=0.7 --shapes=False --seeds=169     \
--trunc-cutoff 28 \
--network=../pretrained_models/next3d_ffhq_512.pkl \
--obj_path=../data/demo/demo.obj    \
--lms_path=../data/demo/demo_kpt2d.txt \
--lms_cond=True --reload_modules=False \
--outdir=/ibex/project/c2241/data/diffusion/w_plus_img_cams_ids_0.7_2m_final_test1 \
--scale_lms=False \
--num_gpus 1 \
--num_writers 10 \
--batch_size 8 \
--sample_cams True \
--sample_ids True \
--dataset_path /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d \
--mesh_path /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d/deca_results_unposed \
--num_samples 1024 \
--horizontal
--lmdb True
```

```bash
python save_samples_next3d.py --outdir=out --trunc=0.7 --shapes=False --seeds=169     \
--trunc-cutoff 28 \
--network=../pretrained_models/next3d_ffhq_512.pkl \
--obj_path=../data/demo/demo.obj    \
--lms_path=../data/demo/demo_kpt2d.txt \
--lms_cond=True --reload_modules=False \
--outdir=/ibex/project/c2241/data/diffusion/w_plus_img_cams_ids_0.7_2m_largefov_largestd_final \
--scale_lms=False \
--num_gpus 4 \
--num_writers 16 \
--batch_size 32 \
--sample_cams True \
--sample_ids True \
--dataset_path /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d \
--mesh_path /ibex/ai/home/parawr/Projects/diffusion/data/ffhq_512_posed_eg3d/deca_results_unposed \
--num_samples 1999872 \
--horizontal_stddev 0.15 \
--vertical_stddev 0.15 \
--lmdb True
```

```
salloc --time 16:00:00 --gres gpu:v100:4 --cpus-per-task 16  --mem 64G --account=conf-cvpr-2023.11.17-wonkap
```

### Reenactment
```bash
python reenact_avatar_next3d.py --drive_root=data/reenactment/obama \
  --network=../pretrained_models/next3d_ffhq_512.pkl \
  --grid=2x1 --seeds=166 --outdir=out --fname=reenact.mp4 \
  --trunc=0.7 --lms_cond=1
```

### Train the diffusion model

```bash
accelerate launch --config_file configs/attempt1.yaml --main_process_port 5431 train_diffusion_eg3d_wplus.py
```

```bash
python src/infra/launch.py slurm=True training.resume=False experiment_name=tinki \
training.workers=4 training.per_gpu_batch_size=32 \
num_gpus=1 sbatch_args.time="01:00:00"  sbatch_args.mem.mem_per_gpu=16 
```

``` bash
python src/infra/launch.py slurm=True training.resume=False \
dataset=partial_2m \
experiment_name=rgb_conditional_seq_v_unet_2m_500k_fulltime_ema_sattn_cattn_no_clip_noautonorm \
model=base  \
diffusion.auto_normalize=False  \
dataset.normalize_w=False
```

```bash
python src/infra/launch.py slurm=False training.resume=False \
dataset=frontal_150k \
experiment_name=rgb_conditional_seq_v_unet_150k_100k_fulltime_ema_sattn_cattn_padinmodel_minmax_styleganviz_deeplab_fixed_npy \
model=base  \
dataset=frontal_150k \
diffusion.auto_normalize=True  \
dataset.normalize_w=True \
dataset.w_norm_type=min_max \
dataset.z_scaler=7.0 \
dataset.padding=[0,0] \
diffusion.objective=pred_v \
num_gpus=4 training.val_freq=5000 \
env=local \
conditioner=deeplab \
training.total_iter=100000 \
training.per_gpu_batch_size=96
```

```bash
python src/infra/launch.py slurm=True \
training.resume=False \
dataset=partial_2m_attr experiment_name=rgb_seg_conditional_seq_v_unet_2m_200k_mlp_embedder \
model=base  \
diffusion.auto_normalize=True  \
dataset.normalize_w=True \
dataset.w_norm_type=min_max \
dataset.z_scaler=7.0 \
dataset.padding=[0,0] \
diffusion.objective=pred_v \
num_gpus=4 \
training.val_freq=5000 \
env=raven \
conditioner=deeplab_seg \
training=embedder
training.total_iter=200000 \
training.per_gpu_batch_size=96 
```
