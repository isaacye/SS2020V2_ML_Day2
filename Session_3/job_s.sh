#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --mem=20000M
#SBATCH --time=0-30:00
#SBATCH --account=def-training-wa
#SBATCH --reservation=snss20_wr_gpu
#SBATCH --output=slurm.%x.%j.out

module load python scipy-stack
source ~/ENV/bin/activate
cd /home/$USER/scratch/$USER/SS2020V2_ML_Day2/Session_3
python /home/$USER/scratch/$USER/SS2020V2_ML_Day2/Session_3/SS20_lab3_LR_MLPg.py
