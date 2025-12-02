#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -J Train_PaiNN
#SBATCH --output=/home/ptim/orcd/scratch/out/Train_PaiNN_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH --mem=10GB
#SBATCH --time=06:00:00

uv run train.py

cat config/settings.py

cat "$0"
