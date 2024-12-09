#!/bin/bash
#SBATCH --output=%x.log
#SBATCH --ntasks=1
#SBATCH --mem=16gb
#SBATCH --time=168:00:00
#SBATCH --cpus-per-task=47
#BATCH --gpus-per-task=1
#BATCH --gres=gpu:1
#SBATCH --partition=main

date;hostname;pwd

srun python run_search.py --transformer-cap $1 --frequency-multiplicator $2 --duration-multiplicator $3 --reward-cfg $4