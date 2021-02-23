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
comet_logger.experiment.log_parameter("timestamp", timestamp)
savedir = "{}/{}".format("/orange/ewhite/b.weinstein/NeonTreeEvaluation/snapshots/",timestamp)

try:
    os.mkdir(savedir)
except:
    pass

#Create objects
m = main.deepforest()
im_callback = images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir, n=10)
m.create_trainer(callbacks=im_callback, logger=comet_logger)

comet_logger.experiment.log_parameters(m.config)
comet_logger.experiment.log_parameters(m.config["train"])
comet_logger.experiment.log_parameters(m.config["validation"])

m.run_train()
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])

torch.save(m.model.state_dict(), "{}/hand_annotated_model.pt".format(savedir))
