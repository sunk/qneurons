#!/bin/bash

#SBATCH --time=47:59:0
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

module load cudnn/v7.6.4-cuda101
module load tensorflow/1.14.0-py36-gpu
module load keras/2.2.5-py36

python runqact.py $@
