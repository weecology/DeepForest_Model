#Test vanilla
import comet_ml
from pytorch_lightning.loggers import CometLogger
from Dead import vanilla
from deepforest import get_data
import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import numpy as np

def test_AliveDeadVanilla():
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
        
    comet_logger.experiment.add_tag("DeadAliveVanilla")    
    
    train_dataset = vanilla.AliveDeadDataset(csv_file="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_train.csv",
                                    root_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB")
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    #Log a few training images
    counter=0        
    for batch in iter(train_dataset):
        if counter > 20:
            break
        image, label = batch 
        image = image.permute(1, 2, 0).numpy()
        comet_logger.experiment.log_image(image, name ="Before Training {} {}".format(label, counter),)
        counter+=1
    
    test_dataset = vanilla.AliveDeadDataset(csv_file="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_test.csv",
                                    root_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB")    

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    trainer = pl.Trainer(fast_dev_run=True, logger=comet_logger)
    
    m = vanilla.AliveDeadVanilla()
    trainer.fit(m, train_dataloader=train_loader, val_dataloaders=test_loader)
    
    true_class, predicted_class = m.dataset_confusion(test_loader)
    comet_logger.experiment.log_confusion_matrix(true_class, predicted_class,labels=["Alive","Dead"])
    
    df = pd.DataFrame({"true_class":np.argmax(true_class,1),"predicted_class":np.argmax(predicted_class,1)})
    true_dead = df[df.true_class == 1]
    dead_recall = true_dead[true_dead.true_class==true_dead.predicted_class].shape[0]/true_dead.shape[0]
    dead_precision = df[df.predicted_class==1].shape[0]/true_dead.shape[0] 
    comet_logger.experiment.log_metric("Dead Recall", dead_recall)
    comet_logger.experiment.log_metric("Dead Predicision", dead_precision)
    
                      