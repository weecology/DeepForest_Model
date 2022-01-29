#Generate point annotations for dead classes
from rasterio.windows import from_bounds
from vanilla import AliveDeadVanilla, get_transform
import start_cluster
import geopandas as gpd
import glob
import pandas as pd
import rasterio as rio
import numpy as np
import os
import re
import torch
from distributed import wait, as_completed

#csv file format is left,bottom,right,top,score,label,height,area,shp_path,geo_index,Year,Site
def get_site(path):
    basename = os.path.basename(path)
    site = re.search("^\d+_(\w+)_\d_", basename).group(1)
    return site


def mine_dead(shp, image_path, model_path, savedir):
    """Apply mining to a single image_path"""
    m = AliveDeadVanilla.load_from_checkpoint(model_path)
    transform = get_transform(augment=False)
    
    if torch.cuda.is_available():
        m = m.eval().cuda(0)
    else:
        m = m.eval()
        
    df = gpd.read_file(shp)
    dead_rows = []
    with rio.open(image_path) as src:
        for index, row in df.iterrows():
            
            left = row.left - 2
            right = row.right + 2
            bottom = row.bottom - 2
            top = row.top + 2
            
            rst = src.read(window = from_bounds(left, bottom, right, top, src.transform))
    
            #preprocess
            try:
                image = transform(np.rollaxis(rst,0,3))
            except Exception:
                continue
            
            if torch.cuda.is_available():
                image = image.cuda(0)     
                prediction = m(image.unsqueeze(0))
                label = np.argmax(prediction.cpu().detach())
            else:
                prediction = m(image.unsqueeze(0))
                label = np.argmax(prediction.cpu().detach())   
            
            if label == 1:
                if prediction.max() > 0.75:
                    dead_rows.append(row)
    
    dead_rows = pd.concat(dead_rows)  
    dead_locations = gpd.GeoDataFrame(dead_rows)
    dead_locations.to_file("dead_locations_OSBS.shp")
    
if __name__ == "__main__":
    client = start_cluster.start(gpus=3)
    shpfiles = glob.glob("/orange/idtrees-collab/draped/*OSBS*.shp")
    rgb_pool = glob.glob("/orange/ewhite/NeonData/OSBS/**/Camera/**/*.tif",recursive=True)
    rgb_dict = {}
    for x in rgb_pool:
        basename = os.path.splitext(os.path.basename(x))[0]
        rgb_dict[basename] = x

    #get one from each site
    site_lists = {}
    for x in shpfiles:
        try:
            site_lists[get_site(x)].append(x)
        except:
            site_lists[get_site(x)] = [x]
            
    futures = []
    for site in site_lists:
        for x in site_lists[site][:2]:
            basename = os.path.splitext(os.path.basename(x))[0]
            future = client.submit(mine_dead,
                          image_path = rgb_dict[basename],
                          shp=x,
                          model_path="/orange/idtrees-collab/DeepTreeAttention/Dead/f4f3664646684a4d9eeff616415960a2.pl",
                          savedir="/orange/idtrees-collab/mining/")
            futures.append(future)
    
    for x in as_completed(futures):
        print(x.result())
    
    