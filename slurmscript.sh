#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -J Train_Strawberry_Intermediate_Aug
#SBATCH --output=/home/ptim/orcd/scratch/out/Train_Strawberry_Intermediate_Aug_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH --mem=20GB
#SBATCH --time=06:00:00

cat config/settings.py

cat "$0"

uv run train.py


