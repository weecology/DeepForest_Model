#srun -p gpu --gpus=1 --mem 10GB --time 5:00:00 --pty -u bash -i
# conda activate deepforest_pytorch
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from deepforest.callbacks import evaluate_callback
from datetime import datetime
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

m = main.deepforest()
trainer = pytorch_lightning.Trainer(logger=comet_logger, max_epochs=1, limit_train_batches=0.01, limit_val_batches=0.01, gpus=m.config["train"]["gpus"])

#Load dataset
train_ds = m.load_dataset(
    csv_file="/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops/hand_annotations_with_header.csv",
    root_dir="/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops/",
    augment=True)

val_ds = m.load_dataset(
    csv_file="/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations_with_header.csv",
    root_dir="/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/",
    augment=True)

trainer.fit(m, train_ds)

precision, recall = m.evaluate("/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations_with_header.csv", iou_threshold=0.4, score_threshold=0.1)

comet_logger.experiment.log_metric(name = "Benchmark precision", value = precision)
comet_logger.experiment.log_metric(name = "Benchmark recall", value = recall)

torch.save(m.backbone.state_dict(), "{}/hand_annotated_model.pt".format(save_dir))
