#!/bin/bash
#SBATCH --job-name=DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

conda activate deepForest_pytorch
cd /home/b.weinstein/DeepForest_Model/

#comet debug
python train.py
