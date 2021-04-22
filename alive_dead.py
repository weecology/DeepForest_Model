#Alive Dead Model, optionally building from release tree crown model
import comet_ml
from datetime import datetime
from deepforest import main
from deepforest import predict
from deepforest import evaluate as evaluate_iou
import glob
import random
import os
import pandas as pd
from pytorch_lightning.loggers import CometLogger
import time
import torch
from torch import optim
from TwoHeadedRetinanet import TwoHeadedRetinanet
from src.predict_second_task import predict_file

#Overwrite default training logs and lr
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
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config["train"]["lr"],
                                   momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=10,
                                                                    verbose=True,
                                                                    threshold=0.0001,
                                                                    threshold_mode='rel',
                                                                    cooldown=0,
                                                                    min_lr=0,
                                                                    eps=1e-08)
        
        #Monitor rate is val data is used
        if self.config["validation"]["csv_file"] is not None:
            return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_classification_task2'}
        else:
            return optimizer
        
    def evaluate_mortality(self,
                 csv_file,
                 root_dir,
                 iou_threshold=None,
                 show_plot=False,
                 savedir=None):
        """Compute intersection-over-union and precision/recall for a given iou_threshold

        Args:
            df: a pandas-type dataframe (geopandas is fine) with columns "name","xmin","ymin","xmax","ymax","label", each box in a row
            root_dir: location of files in the dataframe 'name' column.
            iou_threshold: float [0,1] intersection-over-union union between annotation and prediction to be scored true positive
            show_plot: open a blocking matplotlib window to show plot and annotations, useful for debugging.
            savedir: optional path dir to save evaluation images
        Returns:
            results: dict of ("results", "precision", "recall") for a given threshold
        """
        self.model.eval()

        if not self.device.type == "cpu":
            self.model = self.model.to(self.device)

        predictions = predict_file(model=self.model,
                                           csv_file=csv_file,
                                           root_dir=root_dir,
                                           savedir=savedir,
                                           device=self.device,
                                           iou_threshold=self.config["nms_thresh"])

        predictions["label"] = predictions.label.apply(lambda x: self.numeric_to_label_dict[x])
        ground_df = pd.read_csv(csv_file)

        # if no arg for iou_threshold, set as config
        if iou_threshold is None:
            iou_threshold = self.config["validation"]["iou_threshold"]

        results = evaluate_iou.evaluate(predictions=predictions,
                                        ground_df=ground_df,
                                        root_dir=root_dir,
                                        iou_threshold=iou_threshold,
                                        show_plot=show_plot)

        return results        
        
        

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
    
    #Monkey-patch needed functions to self
    m.topk_candidates = m.model.topk_candidates
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
    
    result_dict = m.evaluate_mortality(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir)
    
    comet_logger.experiment.log_metric("box_precision",result_dict["box_precision"])
    comet_logger.experiment.log_metric("box_recall",result_dict["box_recall"])
    
    result_dict["class_recall"].to_csv("{}/class_recall.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/class_recall.csv".format(savedir))
    
    result_dict["results"].to_csv("{}/results.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/results.csv".format(savedir))
    
    images = glob.glob("{}/*.png".format(savedir))
    random.shuffle(images)
    for img in images[:20]:
        comet_logger.experiment.log_image(img)
        
    #boxes.to_csv("{}/benchmark_predictions.csv".format(savedir))
    #comet_logger.experiment.log_asset("{}/benchmark_predictions.csv".format(savedir))
    
    m.save_model("{}/hand_annotated.pl".format(savedir))
    comet_logger.experiment.log_parameter("saved model", "{}/hand_annotated.pl".format(savedir))

             
if __name__ == "__main__":
    train(train_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv",
          test_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_test.csv")
