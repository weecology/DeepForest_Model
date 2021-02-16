from deepforest import main
from datetime import datetime
from comet_ml import Experiment
import os

comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = "{}/{}".format("/orange/ewhite/b.weinstein/NeonTreeEvaluation/snapshots/",timestamp)
os.mkdir(save_dir)

#Create object
m = main.deepforest()

#Load dataset
m.load_dataset(csv_file="orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops/hand_annotations.csv", train=True)
m.train()
precision, recall = m.evaluate("/orange/b.weinstein/NeonTreeEvaluation/benchmark_annotations.csv", iou_threshold=0.4, probability_threshold=0.2)

comet_experiment.log_metric(name = "Benchmark precision", value = precision)
comet_experiment.log_metric(name = "Benchmark recall", value = recall)

m.save("{}/hand_annotated_model".format(save_dir))