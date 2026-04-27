#!/bin/bash

#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=pace_logs/tapnextpp_%j.log
#SBATCH -N1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=spolisetti6@gatech.edu
#SBATCH --job-name=TapNextpp_testing

source ~/.bashrc
conda activate deformable-rope-dynamics

srun python run_tapnextpp.py

