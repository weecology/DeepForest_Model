#srun -p gpu --gpus=1 --mem 10GB --time 5:00:00 --pty -u bash -i
# conda activate deepforest_pytorch
import comet_ml
from pytorch_lightning.loggers import CometLogger
from deepforest import main
from deepforest.callbacks import images_callback
from datetime import datetime
import os
import time
import random

comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                              project_name="deepforest-pytorch", workspace="bw4sz")
comet_logger.experiment.add_tag("Finetuning")

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
m = main.deepforest.load_from_checkpoint("/orange/ewhite/b.weinstein/retinanet//20210311_185505/pretraining.pl")

im_callback = images_callback(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir, n=3)
m.create_trainer(callbacks=[im_callback], logger=comet_logger)

comet_logger.experiment.log_parameters(m.config)
comet_logger.experiment.log_parameters(m.config["train"])
comet_logger.experiment.log_parameters(m.config["validation"])

m.trainer.fit(m)

result_dict = m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
comet_logger.experiment.log_metric("test_box_precision",result_dict["box_precision"])
comet_logger.experiment.log_metric("test_box_recall",result_dict["box_recall"])

result_dict["class_recall"].to_csv("{}/class_recall.csv".format(savedir))
comet_logger.experiment.log_asset("{}/class_recall.csv".format(savedir))

result_dict["results"].to_csv("{}/results.csv".format(savedir))
comet_logger.experiment.log_asset("{}/results.csv".format(savedir))

boxes = m.predict_file(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
boxes.to_csv("{}/benchmark_predictions.csv".format(savedir))
comet_logger.experiment.log_asset("{}/benchmark_predictions.csv".format(savedir))

m.save_model("{}/hand_annotated.pl".format(savedir))
comet_logger.experiment.log_parameter("saved model", "{}/hand_annotated.pl".format(savedir))

             