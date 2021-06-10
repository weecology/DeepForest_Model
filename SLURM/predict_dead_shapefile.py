#!/bin/bash
#SBATCH --job-name=PredictDeadShapefile   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=3
#SBATCH --mem=20GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepForest_Dead%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepForest_Dead%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

source activate Zooniverse_pytorch

cd /home/b.weinstein/DeepForest_Model/Dead

#comet debug
#python alive_dead.py
python -m cProfile -o predict.prof predict.py
