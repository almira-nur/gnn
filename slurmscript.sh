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

uv add --no-build-isolation torch-scatter

uv run train.py