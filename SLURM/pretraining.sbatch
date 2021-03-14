#!/bin/bash
#SBATCH --job-name=DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=20
#SBATCH --mem=70GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=7

source activate deepforest_pytorch
cd /home/b.weinstein/DeepForest_Model/

#comet debug
python pretraining.py
