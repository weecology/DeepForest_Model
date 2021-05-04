##Vanilla alive dead model
import pandas as pd
import comet_ml
from vanilla import AliveDeadDataset, AliveDeadVanilla
from vanilla import __file__ as ROOT
from predict_field_data import predict_neon
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
import torch
import torch.utils.data as data_utils
import tempfile

ROOT = os.path.dirname(ROOT)

def run(csv_dir = "/orange/idtrees-collab/DeepTreeAttention/data/",
        root_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/",
        savedir="/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/",
        alive_weight=None, gpus=1, num_workers=10, batch_size=32, fast_dev_run=False):    
    
    train_dataset = AliveDeadDataset(csv_file="{}/dead_train.csv".format(csv_dir),
                                    root_dir=root_dir)
    #upsample rare classes
    if alive_weight:
        class_weights = {}
        class_weights[0] = alive_weight
        class_weights[1] = 1
        
        data_weights = []
        for i in range(len(train_dataset)):
            image, label = train_dataset[i]
            data_weights.append(1/class_weights[label])
        
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights = data_weights, num_samples=len(train_dataset))
        shuffle=False
    else:
        sampler = None
        shuffle = True
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler
    )
    
    test_dataset = AliveDeadDataset(csv_file="{}/dead_test.csv".format(csv_dir),
                                    root_dir=root_dir)    
    
    #if debugging, limit the size of the dataset
    if fast_dev_run:
        test_dataset = data_utils.Subset(test_dataset, range(10))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
        
    comet_logger.experiment.add_tag("DeadAliveVanilla")    
    
    #Log a few training images
    counter=0        
    for batch in iter(train_dataset):
        if counter > 20:
            break
        image, label = batch 
        image = image.permute(1, 2, 0).numpy()
        comet_logger.experiment.log_image(image, name ="Before Training {} {}".format(label, counter),)
        counter+=1

    trainer = pl.Trainer(logger=comet_logger, gpus=gpus, max_epochs=60, fast_dev_run=fast_dev_run, checkpoint_callback=False)
    
    m = AliveDeadVanilla()
    trainer.fit(m,train_dataloader=train_loader, val_dataloaders=test_loader)
    
    true_class, predicted_class = m.dataset_confusion(test_loader)
    comet_logger.experiment.log_confusion_matrix(true_class, predicted_class,labels=["Alive","Dead"])
    
    df = pd.DataFrame({"true_class":np.argmax(true_class,1),"predicted_class":np.argmax(predicted_class,1)})
    true_dead = df[df.true_class == 1]
    dead_recall = true_dead[true_dead.true_class==true_dead.predicted_class].shape[0]/true_dead.shape[0]
    if not df[df.predicted_class==1].empty:
        dead_precision = true_dead[true_dead.true_class==true_dead.predicted_class].shape[0]/df[df.predicted_class==1].shape[0]
    else:
        dead_precision = 0
    comet_logger.experiment.log_metric("Dead Recall", dead_recall)
    comet_logger.experiment.log_metric("Dead Precision", dead_precision)    
    
    trainer.save_checkpoint("{}/{}.pl".format(savedir,comet_logger.experiment.get_key()))
    
    #Predict NEON points
    print("Predicting NEON points")
    results = predict_neon(m,
                 boxes_csv="{}/data/trees.csv".format(ROOT),
                 field_path="{}/data/filtered_neon_points.shp".format(ROOT),
                 image_dir=root_dir,
                 savedir=savedir,
                 num_workers=num_workers,
                 debug=fast_dev_run)
    
    results = results.groupby(["plantStatu","Dead"]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0:"count"}).pivot(index="plantStatu",columns="Dead")
    results.to_csv("{}/results.csv".format(tempfile.gettempdir()))
    comet_logger.experiment.log_asset(file_data="{}/results.csv".format(tempfile.gettempdir()), file_name="neon_stems.csv")
    
    if results.shape[0] > 1: 
        results["recall"] = results.apply(lambda x: np.round(x[1]/(x[0]+x[1]) * 100,3), axis=1).fillna(0)
        for index, row in results.iterrows():
            comet_logger.experiment.log_metric(name=index, value=row["recall"])
    

if __name__ == "__main__":
    run(alive_weight=10)
