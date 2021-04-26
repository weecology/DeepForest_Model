#Test vanilla
from Dead import vanilla
from deepforest import get_data
import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch

def test_AliveDeadVanilla():
    train_dataset = vanilla.AliveDeadDataset(csv_file="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_train.csv",
                                    root_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB")
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    
    test_dataset = vanilla.AliveDeadDataset(csv_file="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_test.csv",
                                    root_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB")    


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )


    trainer = pl.Trainer(fast_dev_run=True)
    
    m = vanilla.AliveDeadVanilla()
    trainer.fit(m, train_dataloader=train_loader, val_dataloaders=test_loader)
    
def test_AliveDeadDataset():
    csv_file = get_data("OSBS_029.csv")
    ds = vanilla.AliveDeadDataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file))
    
    df = pd.read_csv(csv_file)
    assert len(ds) == len(df)
        
    for batch in iter(ds):
        box, label = batch  
        plt.imshow(box)
        plt.title('{}'.format(label))
    
    
    