#srun -p gpu --gpus=1 --mem 10GB --time 5:00:00 --pty -u bash -i
# conda activate deepforest_pytorch
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from deepforest.callbacks import images_callback
from datetime import datetime
import torch
import os
import time
import random

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")

#add small sleep for SLURM jobs
time.sleep(random.randint(0,10))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
comet_logger.experiment.log_parameter("timestamp")
save_dir = "{}/{}".format("/orange/ewhite/b.weinstein/NeonTreeEvaluation/snapshots/",timestamp)
try:
    os.mkdir(save_dir)

#Create objects
m = main.deepforest(logger=comet_logger)
im_callback = images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir)
m.create_trainer(callbacks=im_callback)
comet_logger.experiment.log_parameters(m.config["train"])
m.run_train()
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
torch.save(m.backbone.state_dict(), "{}/hand_annotated_model.pt".format(save_dir))
