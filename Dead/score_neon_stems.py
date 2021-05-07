#Plot the 
import cv2
from deepforest.utilities import project_boxes
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import torch
import tempfile
from vanilla import AliveDeadDataset
from vanilla import __file__ as ROOT

ROOT = os.path.dirname(ROOT)

debug = False
boxes_csv="{}/data/trees.csv".format(ROOT)
field_path="{}/data/filtered_neon_points.shp".format(ROOT)
root_dir = "/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB"
savedir = "/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/NEON_points"

boxes = pd.read_csv(boxes_csv)
field = gpd.read_file(field_path)

dataset = AliveDeadDataset(csv_file = boxes_csv, root_dir=root_dir, label_dict={"Tree":0}, train=False, transform=False)

if debug:
    field = field[field.plotID=="SJER_052"]
    boxes = boxes[boxes.image_path=="SJER_052_2018.tif"].reset_index(drop=True) 
    tmp_boxes = "{}/test_box.csv".format(tempfile.gettempdir())
    boxes.to_csv(tmp_boxes)
    dataset = AliveDeadDataset(csv_file = tmp_boxes, root_dir=root_dir, label_dict={"Tree":0}, train=False, transform=False)
    
test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)     

results = []
for name, group in field.groupby("image_path"):
    field_plot = gpd.GeoDataFrame(group, geometry="geometry")
    field_plot.set_crs = group["epsg"]
    field_plot = field_plot[["utmZone","individual","taxonID","siteID","plotID","plantStatu","geometry"]]
    trees = boxes[boxes.image_path == os.path.basename(name)]    
    try:
        trees = project_boxes(trees.copy(), root_dir=root_dir)
    except Exception as e:
        print("{} raised {}".format(name,e))
        continue
    
    df = gpd.sjoin(trees, field_plot)
    results.append(pd.DataFrame(df))

results = pd.concat(results)
    
#Plot standing dead errors
for index in results.index:
    image_array = dataset[index]
    cv2.imwrite("{}/{}.png".format(savedir, results.individual.loc[index]), image_array[:,:,::-1])