#!/bin/bash
#SBATCH --job-name=DeepForest   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_gpu.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_gpu.err
#SBATCH --partition=gpu
#SBATCH --gpus=2

module load tensorflow

export PATH=${PATH}:/home/b.weinstein/miniconda/envs/DeepForest/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda/envs/DeepForest/lib/python3.7/site-packages/
cd /home/b.weinstein/NeonTreeEvaluation_analysis/Weinstein_unpublished/

python train.py
