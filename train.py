#DeepForest 19 site model
from comet_ml import Experiment
import glob
from datetime import datetime
from deepforest import deepforest
from deepforest import tfrecords
from gc import collect

def pretraining(deepforest, BASE_PATH):
    # import comet_ml logger
    comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest", workspace="bw4sz")
        
    comet_experiment.log_parameters(deepforest_model.config)
    comet_experiment.log_parameter("Type","Pretraining")
    comet_experiment.log_parameter("timestamp",timestamp)
    
    list_of_tfrecords = glob.glob(BASE_PATH + "pretraining/tfrecords/*.tfrecord")
    deepforest_model.train(annotations=BASE_PATH + "pretraining/crops/pretraining.csv",
                           input_type="tfrecord",
                           list_of_tfrecords=list_of_tfrecords,
                           comet_experiment=comet_experiment)
    
    if not deepforest_model.config["validation_annotations"] == "None":
        mAP = deepforest_model.evaluate_generator(annotations = deepforest_model.config["validation_annotations"])
        comet_experiment.log_metric("mAP", mAP)
    
    #retrain model based on hand annotation crops, assign the weights from pretraining model, multi-gpu model weights are split.
    deepforest_model.model.save_weights(BASE_PATH + "snapshots/pretraining_weights_{}.h5".format(timestamp))
    deepforest_model.config["weights"] =  BASE_PATH + "snapshots/pretraining_weights_{}.h5".format(timestamp)
    
    return deepforest_model

def finetuning(deepforest_model, BASE_PATH, BENCHMARK_PATH):
    
    input_type = "tfrecord"
    
    #Log parameters
    comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest", workspace="bw4sz")
    
    deepforest_model.config["epochs"] = 100
    comet_experiment.log_parameters(deepforest_model.config)
    comet_experiment.log_parameter("Type","Finetuning")
    comet_experiment.log_parameter("timestamp",timestamp)
    comet_experiment.log_parameter("input_type",input_type)
    
    ##Fine tune model
    list_of_tfrecords = glob.glob(BASE_PATH + "hand_annotations/tfrecords/*.tfrecord")
    deepforest_model.train(annotations=BASE_PATH + "hand_annotations/crops/hand_annotations.csv",
                           input_type=input_type,
                           list_of_tfrecords=list_of_tfrecords,
                           comet_experiment=comet_experiment)
    
    #save weights
    deepforest_model.model.save_weights(BASE_PATH + "snapshots/finetuned_weights_{}.h5".format(timestamp))
    
    #Evaluate benchmark data as generator
    if not deepforest_model.config["validation_annotations"] == "None":
        mAP = deepforest_model.evaluate_generator(annotations = deepforest_model.config["validation_annotations"])
        comet_experiment.log_metric("mAP", mAP)
        
if __name__=="__main__":
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    deepforest_model = deepforest.deepforest()
    
    #Local debug. If False, paths on UF hypergator supercomputing cluster
    DEBUG = False
    
    if DEBUG:
        BASE_PATH = "/Users/ben/Documents/DeepForest_Model/"
        BENCHMARK_PATH = "/Users/ben/Documents/NeonTreeEvaluation/"
        deepforest_model.config["batch_size"] =1        
    else:
        BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"
        BENCHMARK_PATH = "/home/b.weinstein/NeonTreeEvaluation/"    
            
    if not deepforest_model.config["validation_annotations"] == "None":
        deepforest_model.config["validation_annotations"] = BENCHMARK_PATH + deepforest_model.config["validation_annotations"]

    #Run pretraining records
    #deepforest_model = pretraining(deepforest_model, BASE_PATH)
    
    #Optionally set pretraining weights if not running concurrently.
    deepforest_model.config["weights"] = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/snapshots/pretraining_weights_20191110_190026.h5"

    #Fine tune on top of pretraining records
    deepforest_model = finetuning(deepforest_model, BASE_PATH, BENCHMARK_PATH)
