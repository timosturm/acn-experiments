#!/bin/bash
#SBATCH --output=%x.log
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=1
#BATCH --gpus-per-task=1
#BATCH --gres=gpu:1
#SBATCH --partition=main

date;hostname;pwd

srun python $1