#!/bin/bash
#SBATCH --output=%x.log
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=240:00:00
#SBATCH --cpus-per-task=24
#BATCH --gpus-per-task=1
#BATCH --gres=gpu:1
#SBATCH --partition=main

date;hostname;pwd

srun python run_search.py --transformer-cap $1 --reward-cfg $2 --ent-coef $3 --learning-rate $4 --num-envs $5