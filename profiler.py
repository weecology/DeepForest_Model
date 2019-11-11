#Profiling training time
import pandas as pd
import glob

from comet_ml import Experiment
from deepforest import deepforest

#Local debug. If False, paths on UF hypergator supercomputing cluster
DEBUG = False
INPUT_TYPE = "tfrecord" 

if DEBUG:
    BASE_PATH = "/Users/ben/Documents/NeonTreeEvaluation_analysis/Weinstein_unpublished/"
else:
    BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"

##Pretrain deepforest on Silva annotations
deepforest_model = deepforest.deepforest()

# import comet_ml logger
comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest", workspace="bw4sz")
    
comet_experiment.log_parameters(deepforest_model.config)
comet_experiment.log_parameter("Type","Pretraining")

if INPUT_TYPE == "fit_generator":
    comet_experiment.log_parameter("Profiler","fit_generator")
    training_file = pd.read_csv(BASE_PATH + "pretraining/crops/pretraining.csv", names=["image_path","xmin","ymin","xmax","ymax","label"])
    unique_images = training_file.image_path.unique()
    
    sample_data = training_file[training_file.image_path.isin(unique_images[0:10000])]
    
    sample_data.to_csv(BASE_PATH + "pretraining/crops/sample_data.csv", index=False, header=False)
    deepforest_model.train(BASE_PATH + "pretraining/crops/sample_data.csv", comet_experiment=comet_experiment)

if INPUT_TYPE == "tfrecord":
    #Assumes the records have been generated and stored
    comet_experiment.log_parameter("Profiler","tfrecord")
    
    list_of_tfrecords = glob.glob(BASE_PATH + "pretraining/tfrecords/*.tfrecord")
    deepforest_model.train(annotations=BASE_PATH + "pretraining/crops/pretraining.csv",
                           input_type=INPUT_TYPE,
                           list_of_tfrecords=list_of_tfrecords,
                           images_per_epoch = 1156727,
                           comet_experiment=comet_experiment)
        
