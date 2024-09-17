#!/bin/bash
#SBATCH --output=log/%x.log
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=3
#BATCH --gpus-per-task=1
#SBATCH --partition=main

date;hostname;pwd

echo "Starting " + $1

srun python $1
