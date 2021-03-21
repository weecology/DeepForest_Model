#Generate data for model training
from src import crops

#dask integration
from dask_utility import start_dask_cluster

BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"
BENCHMARK_PATH = "/orange/idtrees-collab/NeonTreeEvaluation/"
DATA_PATH = "/orange/ewhite/NeonData/"        

dask_client = start_dask_cluster(number_of_workers=5, mem_size="10GB")
dirname = "hand_annotations/"

#Run Benchmark
#crops.generate_benchmark(BENCHMARK_PATH)
    
#Run pretraining
#crops.generate_pretraining(BASE_PATH, DATA_PATH, BENCHMARK_PATH, dask_client, allow_empty=False)

#Run Training
crops.generate_training(BASE_PATH,BENCHMARK_PATH, dirname, dask_client, allow_empty=False)

