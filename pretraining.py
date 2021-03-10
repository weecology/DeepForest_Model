#srun -p gpu --gpus=1 --mem 10GB --time 5:00:00 --pty -u bash -i
# conda activate deepforest_pytorch
import comet_ml
from datetime import datetime
from deepforest import main
from deepforest.callbacks import images_callback
from deepforest.dataset import get_transform
from deepforest.utilities import collate_fn
import os
from pytorch_lightning.loggers import CometLogger
import random
from src import dataset
import time
import torch

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")

comet_logger.experiment.add_tag("pretraining")

#add small sleep for SLURM jobs
time.sleep(random.randint(0,10))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
comet_logger.experiment.log_parameter("timestamp", timestamp)
savedir = "{}/{}".format("/orange/ewhite/b.weinstein/retinanet/",timestamp)

try:
    os.mkdir(savedir)
except:
    pass

#Create objects
m = main.deepforest()

#override default train loader
train_dataset = dataset.TreeDirectory(
    csv_dir="/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/",
    root_dir="/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/", 
    transforms = get_transform(augment=True))

data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=m.config["batch_size"],
                                          shuffle=True,
                                          collate_fn=collate_fn,
                                          num_workers=m.config["workers"],
                                          )

im_callback = images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir, n=3)
m.create_trainer(callbacks=[im_callback], logger=comet_logger)

comet_logger.experiment.log_parameters(m.config)
comet_logger.experiment.log_parameters(m.config["train"])
comet_logger.experiment.log_parameters(m.config["validation"])

m.trainer.fit(train_dataloader=data_loader)
m.trainer.test(m)
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
boxes = m.predict_file(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
boxes.to_csv("{}/benchmark_predictions.csv".format(savedir))
comet_logger.experiment.log_asset("{}/benchmark_predictions.csv".format(savedir))

m.save_model("{}/pretraining.pl".format(savedir))
comet_logger.experiment.log_parameter("saved model", "{}/pretraining.pl".format(savedir))

