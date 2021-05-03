import numpy as np
import torch
import os
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from deepforest.utilities import project_boxes

from Dead.vanilla import AliveDeadDataset
def predict_neon(dead_model, boxes_csv, field_path_csv, image_dir, savedir, num_workers):
    """For a set of tree predictions, categorize alive/dead and score against NEON field points"""
    boxes = pd.read_csv(boxes_csv)
    field = pd.read_csv(field_path_csv)
        
    if torch.cuda.is_available():
        dead_model = dead_model.to("cuda")
        dead_model.eval()
        
    dataset = AliveDeadDataset(csv_file = boxes_csv, root_dir=image_dir, label_dict={"Tree":0}, train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
        num_workers=num_workers
    )     
    
    gather_predictions = []
    for batch in test_loader:
        if torch.cuda.is_available():
            batch = batch.to("cuda")        
        predictions = dead_model(batch)
        gather_predictions.append(predictions.detach().cpu())

    gather_predictions = np.concatenate(gather_predictions)
    boxes["Dead"] = np.argmax(gather_predictions,1)
    
    results = []
    for name, group in field.groupby("image_path"):
        field_plot = gpd.GeoDataFrame(group, geometry="geometry")
        field_plot.set_crs = group["epsg"]
        field_plot = field_plot[["utmZone","individualID","taxonID","siteID","plotID","plantStatus","geometry"]]
        trees = boxes[boxes.image_path == os.path.basename(name)]    
        try:
            trees = project_boxes(trees, root_dir=image_dir)
        except Exception as e:
            print("{} raised {}".format(name,e))
            continue
        
        df = gpd.sjoin(trees, field_plot)
        results.append(pd.DataFrame(df))
    
    results = pd.concat(results)
        
    return results