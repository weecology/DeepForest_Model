import pandas as pd
import gc
from comet_ml import Experiment
from datetime import datetime

import keras.backend as K
from deepforest import tfrecords
from deepforest import deepforest
from deepforest import utilities

###setup
#Local debug. If False, paths on UF hypergator supercomputing cluster
DEBUG = False

if DEBUG:
    BASE_PATH = "/Users/ben/Documents/NeonTreeEvaluation_analysis/Weinstein_unpublished/"
else:
    BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"
####

#Prepare data

#Load annotations file
annotations = pd.read_csv(BASE_PATH + "pretraining/crops/pretraining.csv", names=["image_path","xmin","ymin","xmax","ymax","label"])

#Select a set of n image
annotations = annotations[annotations.image_path == "2019_DELA_5_423000_3601000_image_0.jpg"].copy()

#Generate tfrecords
annotations_file = BASE_PATH + "pretraining/crops/test.csv"
annotations.to_csv(annotations_file, header=False,index=False)

class_file = utilities.create_classes(annotations_file)

tfrecords_path = tfrecords.create_tfrecords(annotations_file, class_file, size=1)
print("Created {} tfrecords: {}".format(len(tfrecords_path), tfrecords_path))
inputs, targets = tfrecords.create_tensors(tfrecords_path)

#### Fit generator ##
comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest", workspace="bw4sz")


comet_experiment.log_parameter("Type","testing")
comet_experiment.log_parameter("input_type","fit_generator")

#Create model
fitgen_model = deepforest.deepforest()
fitgen_model.config["epochs"] = 1
comet_experiment.log_parameters(fitgen_model.config)

#Train model
fitgen_model.train(annotations_file, input_type="fit_generator", comet_experiment=comet_experiment, images_per_epoch= 1000)

#Evaluate on original annotations
mAP = fitgen_model.evaluate_generator(annotations_file,comet_experiment )
boxes = fitgen_model.prediction_model.predict(inputs, steps=1)

comet_experiment.log_metric("mAP",mAP)

#delete old model to free up space
del fitgen_model
gc.collect()
K.clear_session()

#TFRECORDS
comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest", workspace="bw4sz")

comet_experiment.log_parameter("Type","testing")
comet_experiment.log_parameter("input_type","tfrecord")

#Train model
tfrecord_model = deepforest.deepforest()
tfrecord_model.train(annotations_file, list_of_tfrecords=tfrecords_path, input_type="tfrecord", images_per_epoch= 1000)

#Evaluate on original annotations
mAP = tfrecord_model.evaluate_generator(annotations_file, comet_experiment)
comet_experiment.log_metric("mAP",mAP)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tfrecord_model.model.save_weights(BASE_PATH + "snapshots/finetuned_weights_{}.h5".format(timestamp))

#delete old model to free up space
del tfrecord_model
gc.collect()
K.clear_session()

# from second model https://keras.io/examples/mnist_tfrecord/
test_model = deepforest.deepforest(weights=BASE_PATH + "snapshots/finetuned_weights_{}.h5".format(timestamp))

#Using tfrecord dataset to predict itself
boxes = test_model.prediction_model.predict(inputs, steps=1)

test_mAP = test_model.evaluate_generator(annotations_file, comet_experiment=comet_experiment)
comet_experiment.log_metric("Test Model mAP",test_mAP)
