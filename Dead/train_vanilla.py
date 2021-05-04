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

def index_to_example(index, test_dataset, experiment):
    image_array = test_dataset[index][0].numpy()
    image_array = np.rollaxis(image_array, 0,3)
    image_name = "confusion-matrix-%05d.png" % index
    results = experiment.log_image(
        image_array, name=image_name,
    )
    # Return sample, assetId (index is added automatically)
    return {"sample": image_name, "assetId": results["imageId"]}

def run(csv_dir = "/orange/idtrees-collab/DeepTreeAttention/data/",
        root_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/",
        savedir="/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/",
        alive_weight=None, gpus=1, num_workers=10, batch_size=256, fast_dev_run=False):    
    
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

    trainer = pl.Trainer(logger=comet_logger, gpus=gpus, max_epochs=40, fast_dev_run=fast_dev_run, checkpoint_callback=False)
    
    m = AliveDeadVanilla()
    trainer.fit(m,train_dataloader=train_loader, val_dataloaders=test_loader)
    
    true_class, predicted_class = m.dataset_confusion(test_loader)
    
    comet_logger.experiment.log_confusion_matrix(
        true_class,
        predicted_class,
        labels=["Alive","Dead"], index_to_example_function=index_to_example, test_dataset=test_dataset,
        experiment=comet_logger.experiment)    
    
    
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
    results, box_dataset = predict_neon(m,
                 boxes_csv="{}/data/trees.csv".format(ROOT),
                 field_path="{}/data/filtered_neon_points.shp".format(ROOT),
                 image_dir=root_dir,
                 savedir=savedir,
                 num_workers=num_workers,
                 debug=fast_dev_run)
    
    result_matrix = results.groupby(["plantStatu","Dead"]).apply(lambda x: x.shape[0]).reset_index().rename(columns={0:"count"}).pivot(index="plantStatu",columns="Dead")
    result_matrix.to_csv("{}/results.csv".format(tempfile.gettempdir()))
    comet_logger.experiment.log_asset(file_data="{}/results.csv".format(tempfile.gettempdir()), file_name="neon_stems.csv")
    
    if result_matrix.shape[0] > 1: 
        result_matrix["recall"] = result_matrix.apply(lambda x: np.round(x[1]/(x[0]+x[1]) * 100,3), axis=1).fillna(0)
        for index, row in results.iterrows():
            comet_logger.experiment.log_metric(name=index, value=row["recall"])
    
    #plot the missing standing dead trees
    #standing_dead = results[results.plantStatu=="Standing Dead"]
    for index in results.index:
        image_array = box_dataset[index].numpy()
        image_array = np.rollaxis(image_array, 0,3) 
        comet_logger.experiment.log_image(
            image_data=image_array,
            name="{}_{}:{}".format(results.loc[index].plotID,results.loc[index].plantStatu,results.loc[index].Dead))
        
if __name__ == "__main__":
    run(alive_weight=10)