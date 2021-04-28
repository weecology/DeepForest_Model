#Self-supervised mining for dead trees
from rasterio.windows import from_bounds
from Dead.vanilla import AliveDeadVanilla, get_transform
from Dead import start_cluster
import geopandas as gpd
import glob
import cv2
import rasterio as rio
import numpy as np
import os
import torch

#csv file format is left,bottom,right,top,score,label,height,area,shp_path,geo_index,Year,Site
def mine_dead(shp, image_path, model_path, savedir):
    """Apply mining to a single image_path"""
    m = AliveDeadVanilla()
    m.model.load_state_dict(torch.load(model_path))
    transform = get_transform(augment=False)
    
    if torch.cuda.is_available():
        m = m.eval().cuda(0)
    else:
        m = m.eval()
        
    df = gpd.read_file(shp)
    basename = os.path.splitext(os.path.basename(df.shp_path[0]))[0]
    with rio.open(image_path) as src:
        for index, row in df.iterrows():
            
            left = row.left - 0.5
            right = row.right + 0.5
            bottom = row.bottom + 0.5
            top = row.top - 0.5
            
            rst = src.read(window = from_bounds(left, bottom, right, top, src.transform))
    
            #preprocess
            image = transform(np.rollaxis(rst,0,3))
            
            if torch.cuda.is_available():
                image = image.cuda(0)            
            prediction = m(image.unsqueeze(0))
            label = np.argmax(prediction.detach())
            
            #if Dead, keep
            if label == 1:
                cv2.imwrite("{}/{}_{}.png".format(savedir, basename,index), np.rollaxis(rst,0,3))
                
if __name__ == "__main__":
    client = start_cluster.start(gpus=1)
    shpfiles = glob.glob("/orange/idtrees-collab/draped/*.shp")
    rgb_pool = glob.glob("/orange/ewhite/NeonData/**/Camera/**/*.tif",recursive=True)
    rgb_dict = {}
    for x in rgb_pool:
        basename = os.path.splitext(os.path.basename(x))[0]
        rgb_dict[basename] = x
    
    for x in shpfiles[0]:
        basename = os.path.splitext(os.path.basename(x))[0]
        client.submit(mine_dead,
                      image_path = rgb_dict[basename],
                      shp=x,
                      model_path="/orange/idtrees-collab/DeepTreeAttention/Dead/0259353ec76448b590eec0cb6536734d",
                      savedir="/orange/idtrees-collab/mining/")
    