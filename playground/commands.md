### Generation RGB
Path to obama
/storage/nfs/wamiq/next3d/data/reenactment/obama/01491.obj
/storage/nfs/wamiq/next3d/data/reenactment/obama/01491_kpt2d.txt

```bash
python gen_samples_next3d.py --trunc=0.7 --shapes=true --seeds=169 --network=pretrained_models/next3d_ffhq_512.pkl 
--obj_path=data/demo/demo.obj --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True --outdir=73_samples --shapes=False

python gen_samples_next3d.py --trunc=0.7 --shapes=true --seeds=169 --network=pretrained_models/next3d_ffhq_512.pkl --obj_path=/storage/nfs/wamiq/next3d/data/reenactment/obama/01491.obj --lms_path=/storage/nfs/wamiq/next3d/data/reenactment/obama/01491_kpt2d.txt --lms_cond=True --outdir=73_samples --shapes=False
```

### Generation samples

```bash
python save_samples_next3d.py --outdir=out --trunc=0.7 --shapes=true --seeds=169     --network=pretrained_models/next3d_ffhq_512.pkl --obj_path=data/demo/demo.obj     --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True --only_frontal=True --reload_modules=True --outdir=./data/generated_samples/w_plus_img
```

### Reenactment
```bash
python reenact_avatar_next3d.py --drive_root=data/reenactment/obama \
  --network=pretrained_models/next3d_ffhq_512.pkl \
  --grid=2x1 --seeds=166 --outdir=out --fname=reenact.mp4 \
  --trunc=0.7 --lms_cond=1
```

### Train the diffusion model

```bash
accelerate launch --config_file configs/attempt1.yaml --main_process_port 5431 train_diffusion_eg3d_wplus.py
```