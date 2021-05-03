#Run mined images
from Dead.vanilla import get_transform, AliveDeadVanilla, AliveDeadDataset
import pandas as pd
from pytorch_lightning.loggers import CometLogger
import pytorch_lightning as pl
import numpy as np
from torchvision.datasets import ImageFolder
import torch

def run(checkpoint, annotation_dir, image_dir, csv_dir, savedir, num_workers=10, fast_dev_run=False, batch_size=100, gpus=1):
    m = AliveDeadVanilla.load_from_checkpoint(checkpoint)
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
        
    comet_logger.experiment.add_tag("DeadAlive_Mined")   
    
    transform = get_transform(augment=True)
    train_dataset = ImageFolder(root=annotation_dir, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=False,
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
    
    trainer = pl.Trainer(logger=comet_logger, gpus=gpus, max_epochs=40, fast_dev_run=fast_dev_run)
    
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
    
    trainer.save_checkpoint("Dead/{}.pl".format(comet_logger.experiment.get_key()))    
    
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
