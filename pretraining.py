from deepforest import main
from callbacks import comet_callbacks

from datetime import datetime
from comet_ml import Experiment

comet_experiment = Experiment(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#Create object
m = main.deepforest()

#Load dataset
m.load_dataset(csv_file="orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/pretraining.csv", train=True)

comet_callback = comet_callbacks(experiment = comet_experiment)

m.train(callbacks=comet_callback)
benchmark_mAP, precision, recall = m.evaluate("/orange/b.weinstein/NeonTreeEvaluation/benchmark_annotations.csv", metrics=["mAP","precision","recall"], iou_threshold=0.4, probability_threshold=0.2)

comet_experiment.log_metric(name = "Benchmark mAP", value = benchmark_mAP)
comet_experiment.log_metric(name = "Benchmark precision", value = precision)
comet_experiment.log_metric(name = "Benchmark recall", value = recall)

m.save()