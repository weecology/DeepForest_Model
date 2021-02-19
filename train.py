from deepforest import main
from deepforest.callbacks import evaluate_callback
from datetime import datetime
from comet_ml import CometLogger
import torch
import os
import pytorch_lightning

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_dir = "{}/{}".format("/orange/ewhite/b.weinstein/NeonTreeEvaluation/snapshots/",timestamp)
os.mkdir(save_dir)

#Create objects
eval_callback = evaluate_callback(
    csv_file="/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv", 
    root_dir="/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/",iou_threshold=0.4, score_threshold=0.1)

trainer = pytorch_lightning.Trainer(logger=comet_logger, max_epochs=m.config["train"]["epochs"], callbacks=[evaluate_callback])
m = main.deepforest()

#Load dataset
train_ds = m.load_dataset(
    csv_file="orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops/hand_annotations.csv",
    root_dir="orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops/",
    augment=True)

trainer.fit(m, train_ds)

precision, recall = m.evaluate("/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv", iou_threshold=0.4, score_threshold=0.1)

comet_logger.experiment.log_metric(name = "Benchmark precision", value = precision)
comet_logger.experiment.log_metric(name = "Benchmark recall", value = recall)

torch.save(m.backbone.state_dict(), "{}/hand_annotated_model.pt".format(save_dir))
