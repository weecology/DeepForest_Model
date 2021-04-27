##Vanilla alive dead model
import pandas as pd
import comet_ml
import os
import pytorch_lightning as pl
from skimage import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_lightning.loggers import CometLogger
from torchvision import models, transforms
import matplotlib.pyplot as plt
import torchmetrics


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_transform(augment):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(transforms.Resize([224,224]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)

class AliveDeadDataset(Dataset):

    def __init__(self, csv_file, root_dir, label_dict = {"Alive": 0,"Dead":1}, augment=False):
        """
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        self.transform = get_transform(augment=augment)

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        selected_row = self.annotations.loc[idx]
        img_name = os.path.join(self.root_dir, selected_row["image_path"])
        image = io.imread(img_name)

        # select annotations
        xmin, xmax, ymin, ymax = selected_row[["xmin","xmax","ymin","ymax"]].values.astype(int)
        
        xmin = np.max([0,xmin-5])
        xmax = np.min([image.shape[1],xmax+5])
        ymin = np.max([0,ymin-5])
        ymax = np.min([image.shape[0],ymax+5])
        
        box = image[ymin:ymax, xmin:xmax]
        
        # Labels need to be encoded
        label = self.label_dict[selected_row.label]
        box = self.transform(box)

        return box, label
    
    
#Lightning Model
class AliveDeadVanilla(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)        
        self.accuracy = torchmetrics.Accuracy(multiclass=True)        

    def forward(self, x):
        output = self.model(x)
        pred = F.softmax(output)
        
        return pred
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs,y)
        self.log("train_loss",loss)
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs,y)
        self.log("val_loss",loss)        
        self.accuracy(outputs, y)
 
    def validation_epoch_end(self, outputs):
        self.log('val_acc', self.accuracy.compute())
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer
    
    def dataset_confusion(self, loader):
        """Create a confusion matrix from a data loader"""
        true_class = []
        predicted_class = []
        for batch in loader:
            x,y = batch
            true_class.append(F.one_hot(y,num_classes=2).detach().numpy())
            prediction = self(x)
            predicted_class.append(prediction.detach().numpy())
        
        true_class = np.concatenate(true_class)
        predicted_class = np.concatenate(predicted_class)

        return true_class, predicted_class
            
if __name__ == "__main__":
    #create train loader
    
    train_dataset = AliveDeadDataset(csv_file="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv",
                                    root_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/")
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=5
    )
    
    test_dataset = AliveDeadDataset(csv_file="/orange/idtrees-collab/DeepTreeAttention/data/dead_test.csv",
                                    root_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/")    

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=5
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

    trainer = pl.Trainer(logger=comet_logger, gpus=1, max_epochs=20)
    
    m = AliveDeadVanilla()
    trainer.fit(m,train_dataloader=train_loader, val_dataloaders=test_loader)
    
    true_class, predicted_class = m.dataset_confusion(test_loader)
    comet_logger.experiment.log_confusion_matrix(true_class, predicted_class,labels=["Alive","Dead"])
    
    df = pd.DataFrame({"true_class":np.argmax(true_class,1),"predicted_class":np.argmax(predicted_class,1)})
    true_dead = df[df.true_class == 1]
    dead_recall = true_dead[true_dead.true_class==true_dead.predicted_class].shape[0]/true_dead.shape[0]
    dead_precision = true_dead[true_dead.true_class==true_dead.predicted_class].shape[0]/df[df.predicted_class==1].shape[0]
    comet_logger.experiment.log_metric("Dead Recall", dead_recall)
    comet_logger.experiment.log_metric("Dead Precision", dead_precision)    
    
