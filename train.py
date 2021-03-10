#srun -p gpu --gpus=1 --mem 10GB --time 5:00:00 --pty -u bash -i
# conda activate deepforest_pytorch
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from deepforest import get_data
from deepforest.callbacks import images_callback
from datetime import datetime
import os
import time
import random

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")
comet_logger.experiment.add_tag("training")

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

im_callback = images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir, n=5)
m.create_trainer(callbacks=[im_callback], logger=comet_logger)
m.trainer.checkpoint_callback = True

comet_logger.experiment.log_parameters(m.config)
comet_logger.experiment.log_parameters(m.config["train"])
comet_logger.experiment.log_parameters(m.config["validation"])

m.trainer.fit(m)
m.trainer.test(m)
m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
boxes = m.predict_file(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
boxes.to_csv("{}/benchmark_predictions.csv".format(savedir))
comet_logger.experiment.log_asset("{}/benchmark_predictions.csv".format(savedir))

m.save_model("{}/hand_annotated.pl".format(savedir))
comet_logger.experiment.log_parameter("saved model", "{}/hand_annotated.pl".format(savedir))

             
