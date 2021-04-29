

from src import dataset
from deepforest.dataset import get_transform
from deepforest import main
from deepforest.utilities import collate_fn
import torch

m = main.deepforest()

train_dataset = dataset.TreeDirectory(
    csv_dir="/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/",
    root_dir="/orange/ewhite/b.weinstein/NeonTreeEvaluation/pretraining/crops/", 
    transforms = get_transform(augment=True))

data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=m.config["batch_size"],
                                          shuffle=True,
                                          collate_fn=collate_fn,
                                          num_workers=m.config["workers"],
                                          )    

for x in data_loader:
    path, image, targets = x
    print(path)
    