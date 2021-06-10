#!/bin/bash
#SBATCH --job-name=DeepForest_generate   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ranks
#SBATCH --mem=20GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/generate.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/generate.err

source activate Zooniverse_pytorch
cd /home/b.weinstein/DeepForest_Model/

python generate.py
