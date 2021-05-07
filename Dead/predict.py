#Predict tile
from deepforest import main
from deepforest.utilities import project_boxes
from glob import glob
import geopandas as gpd
import numpy as np
import torch
import tempfile
import os
import math
import fiona
import rasterio as rio
from rasterio.mask import mask
import pandas as pd
from vanilla import AliveDeadDataset, AliveDeadVanilla

def bounds_to_geoindex(bounds):
    """Convert an extent into NEONs naming schema
    Args:
        bounds: list of top, left, bottom, right bounds, usually from geopandas.total_bounds
    Return:
        geoindex: str {easting}_{northing}
    """
    easting = min(bounds[0], bounds[2])
    northing = min(bounds[1], bounds[3])

    easting = math.floor(easting / 1000) * 1000
    northing = math.floor(northing / 1000) * 1000

    geoindex = "{}_{}".format(easting, northing)

    return geoindex

def predict_trees(tree_detector, tile_path):
    """Predict tree bounding boxes and return a csv"""
    trees = tree_detector.predict_tile(tile_path)
    trees["image_path"] = os.path.basename(tile_path)
    tree_csv = "{}/{}.csv".format(tempfile.gettempdir(), os.path.splitext(os.path.basename(tile_path))[0])
    trees.to_csv(tree_csv)
    
    return tree_csv
    
def predict_dead(dead_model, tree_csv, root_dir, batch_size=100, num_workers=5):
    """For a given model and csv file of bounding box detections, predict alive dead class"""
    boxes = pd.read_csv(tree_csv)
    dataset = AliveDeadDataset(csv_file = tree_csv, root_dir=root_dir, train=False)
    
    if torch.cuda.is_available():
        dead_model = dead_model.to("cuda")
        dead_model.eval()
    
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
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
    
    return boxes

def create_tiles(shp, image_pool, savedir):
    """For a given shapefile, lookup the geoindex and crop a numpy array for each array
    
    Returns:
        tile_paths: list of cropped tile paths of RGB data
    """
    df = gpd.read_file(shp)
    bounds = df.total_bounds
    geo_index = bounds_to_geoindex(bounds)
    
    #pad geoindex by 1 in each direction to get all surrounding tiles
    tiles = [x for x in image_pool if geo_index in x]  
    
    if len(tiles) == 0:
        raise IOError("No tiles found for geoindex {}".format(geo_index))
    
    tile_paths = []
    with fiona.open(shp, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        
        for x in tiles:
            with rio.open(x) as src:
                out_image, out_transform = mask(src, shapes, crop=True)
                out_meta = src.meta
                out_meta.update({"driver": "GTiff",
                                 "height": out_image.shape[1],
                                 "width": out_image.shape[2],
                                 "transform": out_transform})
            
                crop_path = "{}/{}".format(savedir,os.path.basename(x))
                tile_paths.append(crop_path)
                with rio.open(crop_path, "w", **out_meta) as dest:
                    dest.write(out_image)
    return tile_paths
    
def run(checkpoint_path, image_glob, shape_dir, savedir, num_workers=5):
    dead_model = AliveDeadVanilla.load_from_checkpoint(checkpoint_path)
    
    if torch.cuda.is_available():
        dead_model = dead_model.to("cuda")
        dead_model.eval()
        
    shps= glob("{}/*.shp".format(shape_dir))
    
    tree_detector = main.deepforest()
    tree_detector.use_release()
    
    image_pool = glob(image_glob, recursive=True)
    if len(image_pool) == 0:
        raise IOError("No images found in image_glob {}".format(image_glob))
    
    for shp in shps:
        tile_paths = create_tiles(shp, image_pool, savedir)
        for tile_path in tile_paths:    
            trees_csv = predict_trees(tree_detector, tile_path)
            dead_trees = predict_dead(dead_model, trees_csv, root_dir=os.path.dirname(tile_path), num_workers=num_workers)
            projected_trees = project_boxes(df=dead_trees, root_dir=os.path.dirname(tile_path))
            basename = os.path.basename(os.path.splitext(tile_path)[0])
            projected_trees.to_file("{}/{}.shp".format(savedir, basename))
    

if __name__ == "__main__":
    run(
        checkpoint_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/f759259caad04bbc98c013643b2fbdae.pl",
        image_glob="/orange/ewhite/NeonData/**/Camera/**/*.tif",
        shape_dir="/orange/ewhite/b.weinstein/DeadTrees/site_shps/",
        savedir="/orange/ewhite/b.weinstein/DeadTrees/predictions")