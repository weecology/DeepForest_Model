#Alive Dead Model, optionally building from release tree crown model
import comet_ml
import glob
import time
import random
from datetime import datetime
import os
from deepforest import main
from pytorch_lightning.loggers import CometLogger
from TwoHeadedRetinanet import TwoHeadedRetinanet


#Overwrite default training log
class alive_dead_module(main.deepforest):
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        path, images, targets = batch
    
        loss_dict = self.model.forward(images, targets)
    
        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])
        # Log loss
        for key, value in loss_dict.items():
            self.log("train_{}".format(key), value, on_epoch=True)
            
        return losses

def train(train_path, test_path, pretrained=False, image_dir = "/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/", debug=False, savedir="/orange/idtrees-collab/DeepTreeAttention/Dead/"):
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
    comet_logger.experiment.add_tag("DeadAlive")
    
    
    #add small sleep for SLURM jobs
    time.sleep(random.randint(0,10))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comet_logger.experiment.log_parameter("timestamp", timestamp)
    savedir = "{}/{}".format(savedir,timestamp)
    
    try:
        os.mkdir(savedir)
    except:
        pass
    
    #Get release state dict
    m = alive_dead_module()
    m.use_release()
    
    #Overwrite original retinanet with a two headed task
    m.model = TwoHeadedRetinanet(trained_model=m.model, num_classes_task2=2, freeze_original=True)
    m.label_dict = {"Alive":0,"Dead":1}
    #update the labels for the new task
    
    m.config["train"]["csv_file"] = train_path
    m.config["train"]["root_dir"] = image_dir
    m.config["validation"]["csv_file"] = test_path
    m.config["validation"]["root_dir"] = image_dir
    
    if debug:
        m.config["train"]["fast_dev_run"] = True
        m.config["gpus"] = None
        m.config["workers"] = 0
        m.config["distributed_backend"] = None
        m.config["batch_size"] = 2
    
    comet_logger.experiment.log_parameters(m.config)
    comet_logger.experiment.log_parameters(m.config["train"])
    comet_logger.experiment.log_parameters(m.config["validation"])
    
    m.create_trainer(logger=comet_logger)
    
    m.trainer.fit(m)
    
    result_dict = m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
    
    comet_logger.experiment.log_metric("box_precision",result_dict["box_precision"])
    comet_logger.experiment.log_metric("box_recall",result_dict["box_recall"])
    
    result_dict["class_recall"].to_csv("{}/class_recall.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/class_recall.csv".format(savedir))
    
    result_dict["results"].to_csv("{}/results.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/results.csv".format(savedir))
    
    boxes = m.predict_file(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir)
    images = glob.glob("{}/*.png".format(savedir))
    random.shuffle(images)
    for img in images[:20]:
        comet_logger.experiment.log_image(img)
    boxes.to_csv("{}/benchmark_predictions.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/benchmark_predictions.csv".format(savedir))
    
    m.save_model("{}/hand_annotated.pl".format(savedir))
    comet_logger.experiment.log_parameter("saved model", "{}/hand_annotated.pl".format(savedir))

             
if __name__ == "__main__":
    train(train_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv",
          test_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_test.csv")