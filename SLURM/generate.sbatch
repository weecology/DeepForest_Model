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

module load tensorflow/1.14.0

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/crowns/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda3/envs/crowns/lib/python3.7/site-packages/
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/crowns/lib/:${LD_LIBRARY_PATH}

cd /home/b.weinstein/DeepForest_Model/

python generate.py
python GenerateAnchors.py
