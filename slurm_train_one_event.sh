#!/bin/bash
#SBATCH --job-name=dune_one_event
#SBATCH --output=logs/dune_one_event_%j.out
#SBATCH --error=logs/dune_one_event_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -e

echo "Job started on:"
hostname
date

cd /gpfs/workdir/thibauts/GUNTAM

module load anaconda3/2023.09-0/none-none
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate /gpfs/workdir/thibauts/conda_envs/guntam_v100

echo "Python:"
which python

echo "CUDA visible devices:"
echo $CUDA_VISIBLE_DEVICES

python GUNTAM/Seed/train_dune_full_one_event.py

echo "Job finished:"
date
