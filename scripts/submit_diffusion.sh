#!/bin/bash --login
#SBATCH -N 1
#SBATCH --export=ALL,NCCL_SOCKET_IFNAME=eth0
#SBATCH -J prim_final
#SBATCH -o output/slurm/%J.out
#SBATCH -e output/slurm/%J.err
#SBATCH --time=36:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:v100:8

conda activate faclab

python train.py --enc_layer 24 --distr True --fp16 True --num_workers 24 --dim 528 --bs 544 --n_heads 12 --notes full_tokens
