#Test dataloader
import os
import pytest
from src import dataset
from deepforest.dataset import get_transform
from deepforest import get_data
import pandas as pd

def test_TreeDirectory(tmpdir):
    """Create two sample files in a directory"""
    csv_file = get_data("OSBS_029.csv")
    sample_file = pd.read_csv(csv_file)
    sample_file.to_csv("{}/1.csv".format(tmpdir))
    sample_file.to_csv("{}/2.csv".format(tmpdir))    
    ds = dataset.TreeDirectory(csv_dir=tmpdir, root_dir=os.path.dirname(csv_file), transforms=get_transform(augment=True))
    assert len(ds) == 2
    
    batch = next(iter(ds))
    assert len(batch) == 3
    
    path, data, targets = batch
    assert targets["boxes"].shape[0] == sample_file.shape[0]
    
    
