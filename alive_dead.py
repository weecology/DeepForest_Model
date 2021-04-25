#Alive Dead Model, optionally building from release tree crown model
import comet_ml
import copy
from datetime import datetime
from deepforest import main
from deepforest import predict
from deepforest import visualize
from deepforest import evaluate as evaluate_iou
import glob
import random
import numpy as np
import os
import pandas as pd
from pytorch_lightning.loggers import CometLogger
import time
import torch
import tempfile
from torch import optim
from TwoHeadedRetinanet import TwoHeadedRetinanet
from src.predict_second_task import predict_file

def view_training(paths):
    """For each site, grab three images and view annotations"""
    m = main.deepforest(num_classes=2, label_dict = {"Dead":0,"Alive":1})
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
    
    comet_logger.experiment.add_tag("view_training")
    for x in paths:
        ds = m.load_dataset(csv_file=x, root_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/", shuffle=True)
        for i in iter(ds):
            image_path, image, targets = i
            df = visualize.format_boxes(targets[0], scores=False)
            image = np.moveaxis(image[0].numpy(),0,2)
            plot, ax = visualize.plot_predictions(image, df)
            with tempfile.TemporaryDirectory() as tmpdirname:
                plot.savefig("{}/{}".format(tmpdirname, image_path[0]), dpi=300)
                comet_logger.experiment.log_image("{}/{}".format(tmpdirname, image_path[0]))                

def assert_state_dict_not_equal(model_1, model_2):
    """Assert that two pytorch model state dicts are identical
    from https://discuss.pytorch.org/t/two-models-with-same-weights-different-results/8918/7
    Args:
        model_1: a state_dict object from a model
        model_2: a state_dict object from a 2nd model
    Returns:
        None: assertion that models are the same
    """
    models_differ = 0    
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    assert not models_differ == 0
    
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
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset

        """
        path, images, targets = batch

        #Get losses
        self.model.train()
        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        # Log loss
        for key, value in loss_dict.items():
            self.log("val_{}".format(key), value, on_epoch=True)

        self.model.eval()
        predictions = self.model.forward(images)
        
        dead_trees = 0
        for x in predictions:
            dead_trees+=sum(x["labels_task2"] == 0)
        
        return losses, dead_trees
    
    def validation_epoch_end(self, outputs):
        dead_trees = torch.stack([x[1] for x in outputs]).sum()
        self.log("dead_trees", dead_trees)
        
    
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
    
    #Overwrite original retinanet with a two headed task, remake the label dictionary
    m.model = TwoHeadedRetinanet(trained_model=m.model, num_classes_task2=2, freeze_original=True)
    m.label_dict = {"Dead":0,"Alive":1}
    m.numeric_to_label_dict = {v: k for k, v in m.label_dict.items()}
    
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
    
    #Assert that new regression head is training
    original_state = copy.deepcopy(m.model.head.classification_head_task2.state_dict())
    m.trainer.fit(m)
    trained_state = m.model.head.classification_head_task2.state_dict()
    
    assert_state_dict_not_equal(model_1=original_state, model_2=trained_state)
    
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
    
    try:
        m.save_model("{}/hand_annotated.pl".format(savedir))
        comet_logger.experiment.log_parameter("saved model", "{}/hand_annotated.pl".format(savedir))
    except:
        pass
    
    return m

             
if __name__ == "__main__":
    view_training(paths=["/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv", "/orange/idtrees-collab/DeepTreeAttention/data/dead_test.csv"])
    train(train_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv",
          test_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_test.csv")
