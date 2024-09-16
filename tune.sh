#!/bin/bash
#SBATCH --output=log/%x.log
#SBATCH --ntasks=ranking
#SBATCH --mem=80gb
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=64
#BATCH --gpus-per-task=ranking
#SBATCH --partition=main

date;hostname;pwd

echo "Starting " + $1

srun python $1
