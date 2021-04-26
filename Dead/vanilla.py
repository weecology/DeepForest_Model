##Vanilla alive dead model
import os
import pytorch_lightning as pl
import pandas as pd
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_lightning.loggers import CometLogger
from torchvision import models, transforms
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_transform(augment):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(transforms.Resize(224))
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

        box = image[selected_row.xmin:selected_row.xmax,selected_row.ymin:selected_row.ymax]
        
        # Labels need to be encoded
        label = selected_row.label.apply(
            lambda x: self.label_dict[x]).values.astype(int)
        
        box = self.transform(box)

        return box, label
    
    
#Lightning Model
class AliveDeadVanilla(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18()

    def forward(self, x):
        model_ft = self.resnet18(x)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        
        outputs = self.forward(x)
        loss = nn.CrossEntropyLoss(outputs,y)
        
        self.log(loss)
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.forward(x)
        loss = nn.CrossEntropyLoss(outputs,y)
        
        self.log("val_loss",loss)        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer        
    
if __name__ == "__main__":
    #create train loader
    
    train_loader = AliveDeadDataset(csv_file="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv",
                                    root_dir="/orange/idtrees-collab/DeepTreeAttention/data/")
    
    test_loader = AliveDeadDataset(csv_file="/orange/idtrees-collab/DeepTreeAttention/data/dead_test.csv",
                                    root_dir="/orange/idtrees-collab/DeepTreeAttention/data/")    
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
        
    comet_logger.experiment.add_tag("DeadAliveVanilla")    
    trainer = pl.Trainer(logger=comet_logger)
    
    m = AliveDeadVanilla()
    trainer.fit(train_dataloader=train_loader, val_dataloaders=test_loader)
    for batch in test_loader:
        x,y = batch
        