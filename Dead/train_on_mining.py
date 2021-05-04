#Run mined images
from vanilla import get_transform, AliveDeadVanilla, AliveDeadDataset
import os
import pandas as pd
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as pl
from predict_field_data import predict_neon
import numpy as np
from torchvision.datasets import ImageFolder
import torch
import tempfile

from vanilla import __file__ as ROOT
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

def run(checkpoint, annotation_dir, image_dir, csv_dir, savedir, num_workers=10, fast_dev_run=False, batch_size=256, gpus=1):
    m = AliveDeadVanilla.load_from_checkpoint(checkpoint)
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
        
    comet_logger.experiment.add_tag("DeadAlive_Mined")   
    
    transform = get_transform(augment=True)
    train_dataset = ImageFolder(root=annotation_dir, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )    
    
    test_dataset = AliveDeadDataset(csv_file="{}/dead_test.csv".format(csv_dir),
                                    root_dir=image_dir)    
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    trainer = pl.Trainer(logger=comet_logger, gpus=gpus, max_epochs=40, fast_dev_run=fast_dev_run, checkpoint_callback=False)
    
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
    
    trainer.save_checkpoint("Dead/{}.pl".format(comet_logger.experiment.get_key()))    
    
    #Predict NEON points
    print("Predicting NEON points")
    results, box_dataset = predict_neon(m,
                 boxes_csv="{}/data/trees.csv".format(ROOT),
                 field_path="{}/data/filtered_neon_points.shp".format(ROOT),
                 image_dir=image_dir,
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
    
    #Plot standing dead errors
    for index in results.index:
        image_array = box_dataset[index].numpy()
        image_array = np.rollaxis(image_array, 0,3) 
        comet_logger.experiment.log_image(
            image_data=image_array,
            name="{}_{}:{}".format(results.loc[index].plotID,results.loc[index].plantStatu,results.loc[index].Dead))    
    
if __name__ == "__main__":
    run(
        checkpoint="/orange/idtrees-collab/DeepTreeAttention/Dead/cef3e91d8a9c4e848d85d333233b3c7f.pl",
        annotation_dir="/orange/idtrees-collab/DeepTreeAttention/Dead/annotations/",
        csv_dir="/orange/idtrees-collab/DeepTreeAttention/data/",
        image_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB",
        savedir="/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots",
        fast_dev_run=False,
        gpus=1,
        num_workers=10)
