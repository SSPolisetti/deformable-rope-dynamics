#!/bin/bash

#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=pace_logs/WLLearning_%j.log
#SBATCH -N1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=spolisetti6@gatech.edu
#SBATCH --job-name=WLLearning

source ~/.bashrc

conda activate deformable_rope_sim
python 
