#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ms_model_train.py
#SBATCH --mem=800
 
module purge
module load Python/3.7.4-GCCcore-8.3.0

source .envs/py374_env1/bin/activate

python ms_model_train.py
