#Compare alive/dead predictions to NEON field data
import vanilla
import os
import pandas as pd
import torch
from deepforest import main
from deepforest.utilities import project_boxes

import matplotlib as plt
import numpy as np
import geopandas as gpd
import seaborn as sns
from shapely.geometry import Point
import rasterstats
import re
import glob
import math
from pyproj import CRS


def find_image(plotID, image_pool):
    try:
        return [x for x in image_pool if plotID in x][-1]
    except:
        return None
    
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

def find_sensor_path(lookup_pool, shapefile=None, bounds=None):
    """Find a hyperspec path based on the shapefile using NEONs schema
    Args:
        bounds: Optional: list of top, left, bottom, right bounds, usually from geopandas.total_bounds. Instead of providing a shapefile
        lookup_pool: glob string to search for matching files for geoindex
    Returns:
        year_match: full path to sensor tile
    """

    if shapefile is None:
        geo_index = bounds_to_geoindex(bounds=bounds)
        match = [x for x in lookup_pool if geo_index in x]
        match.sort()
        try:
            year_match = match[-1]
        except Exception as e:
            raise ValueError("No matches for geoindex {} in sensor pool".format(geo_index))
    else:

        #Get file metadata from name string
        basename = os.path.splitext(os.path.basename(shapefile))[0]
        geo_index = re.search("(\d+_\d+)_image", basename).group(1)
        match = [x for x in lookup_pool if geo_index in x]
        match.sort()
        try:
            year_match = match[-1]
        except Exception as e:
            raise ValueError("No matches for geoindex {} in sensor pool".format(geo_index))

    return year_match

def non_zero_99_quantile(x):
    """Get height quantile of all cells that are no zero"""
    mdata = np.ma.masked_where(x < 0.5, x)
    mdata = np.ma.filled(mdata, np.nan)
    percentile = np.nanpercentile(mdata, 99)
    return (percentile)

def postprocess_CHM(df, lookup_pool):
    """Field measured height must be within min_diff meters of canopy model"""
    #Extract zonal stats
    try:
        CHM_path = find_sensor_path(lookup_pool=lookup_pool, bounds=df.total_bounds)
    except Exception as e:
        raise ValueError("Cannot find CHM path for {} from plot {} in lookup_pool: {}".format(df.total_bounds, df.plotID.unique(),e))
    draped_boxes = rasterstats.zonal_stats(df.geometry.__geo_interface__,
                                           CHM_path,
                                           add_stats={'q99': non_zero_99_quantile})
    df["CHM_height"] = [x["q99"] for x in draped_boxes]

    #if height is null, assign it
    df.height.fillna(df["CHM_height"], inplace=True)
        
    return df

        
def filter_CHM(shp, lookup_glob):
        """For each plotID extract the heights from LiDAR derived CHM
        Args:
            shp: shapefile of data to filter
            lookup_glob: recursive glob search for CHM files
        """    
        filtered_results = []
        lookup_pool = glob.glob(lookup_glob, recursive=True)        
        for name, group in shp.groupby("plotID"):
            try:
                result = postprocess_CHM(group, lookup_pool=lookup_pool)
                filtered_results.append(result)
            except Exception as e:
                print("plotID {} raised: {}".format(name,e))
                
        filtered_shp = gpd.GeoDataFrame(pd.concat(filtered_results,ignore_index=True))
        
        return filtered_shp

def utm_to_epsg(utm_zone):
    # use PROJ string, assuming a default WGS84
    crs = CRS.from_string('+proj=utm +zone={} +north'.format(utm_zone))
    return crs.to_authority()[0] 
    
def load_field_data(field_path, debug=False):
    field = pd.read_csv(field_path)
    if debug:
        field = field[field.plotID == "SJER_052"]
        
    field = field[~field.elevation.isnull()]
    field = field[~field.growthForm.isin(["liana","small shrub"])]
    field = field[~field.growthForm.isnull()]
    field = field[~field.plantStatus.isnull()]        
    
    groups = field.groupby("individualID")
    shaded_ids = []
    for name, group in groups:
        shaded = any([x in ["Full shade", "Mostly shaded"] for x in group.canopyPosition.values])
        if shaded:
            if any([x in ["Open grown", "Full sun"] for x in group.canopyPosition.values]):
                continue
            else:
                shaded_ids.append(group.individualID.unique()[0])
        
    field = field[~(field.individualID.isin(shaded_ids))]
    field = field[(field.height > 3) | (field.height.isnull())]
    field = field[field.stemDiameter > 10]
    field = field[~field.taxonID.isin(["BETUL", "FRAXI", "HALES", "PICEA", "PINUS", "QUERC", "ULMUS", "2PLANT"])]
    field = field[~(field.eventID.str.contains("2014"))]
    with_heights = field[~field.height.isnull()]
    with_heights = with_heights.loc[with_heights.groupby('individualID')['height'].idxmax()]
    
    missing_heights = field[field.height.isnull()]
    missing_heights = missing_heights[~missing_heights.individualID.isin(with_heights.individualID)]
    missing_heights = missing_heights.groupby("individualID").apply(lambda x: x.sort_values(["eventID"],ascending=False).head(1)).reset_index(drop=True)
  
    field = pd.concat([with_heights,missing_heights])
    
    #remove multibole
    field = field[~(field.individualID.str.contains('[A-Z]$',regex=True))]

    #List of hand cleaned errors
    known_errors = ["NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03422","NEON.PLA.D03.OSBS.03382", "NEON.PLA.D17.TEAK.01883"]
    field = field[~(field.individualID.isin(known_errors))]
    field = field[~(field.plotID == "SOAP_054")]
    
    #Create shapefile
    field["geometry"] = [Point(x,y) for x,y in zip(field["itcEasting"], field["itcNorthing"])]
    shp = gpd.GeoDataFrame(field)
    
    #HOTFIX, BLAN has some data in 18N UTM, reproject to 17N update columns
    BLAN_errors = shp[(shp.siteID == "BLAN") & (shp.utmZone == "18N")]
    BLAN_errors.set_crs(epsg=32618, inplace=True)
    BLAN_errors.to_crs(32617,inplace=True)
    BLAN_errors["utmZone"] = "17N"
    BLAN_errors["itcEasting"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][0])
    BLAN_errors["itcNorthing"] = BLAN_errors.geometry.apply(lambda x: x.coords[0][1])
    
    #reupdate
    shp.loc[BLAN_errors.index] = BLAN_errors
    
    #Oak Right Lab has no AOP data
    shp = shp[~(shp.siteID.isin(["PUUM","ORNL"]))]
    shp["epsg"] = shp.utmZone.apply(lambda x: utm_to_epsg(x))
    
    return shp

def predict_trees(tree_detector, image_paths):
    """Predict alive dead for each crop in the field data"""
    #Predict tree location
    predictions = []
    for image_path in image_paths:
        boxes = tree_detector.predict_image(path=image_path)
        if boxes is not None:
            boxes["image_path"] = os.path.basename(image_path)
            predictions.append(boxes)
    predictions = pd.concat(predictions)
    
    return predictions

def run(checkpoint_path, image_dir, savedir, field_path, num_workers=10, canopy_filter=False, debug=False):
    """For each field plot, predict the location of trees and their alive dead class, join to the spatial data and compare to field tag"""
    dead_model = vanilla.AliveDeadVanilla.load_from_checkpoint(checkpoint_path)
    
    if torch.cuda.is_available():
        dead_model = dead_model.to("cuda")
        dead_model.eval()
    
    tree_detector = main.deepforest()
    tree_detector.use_release()
    
    field = load_field_data(field_path, debug=debug)
    print("loaded field data")
    if debug:
        field = field[field.plotID=="SJER_052"]
        
    image_pool = glob.glob("{}/*.tif".format(image_dir))
    
    #Find image locations
    field["image_path"] = field.plotID.apply(lambda plotID: find_image(plotID,image_pool))
    field = field[field.image_path.notnull()]
    
    image_paths =  field.image_path.unique()
    boxes = predict_trees(tree_detector, image_paths)
    boxes.to_csv("{}/trees.csv".format(savedir))
    print("trees predicted")
    if canopy_filter:
        lookup_glob = "/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif"
        field = filter_CHM(field, lookup_glob)
    
    dataset = vanilla.AliveDeadDataset(csv_file = "{}/trees.csv".format(savedir), root_dir=image_dir, label_dict={"Tree":0}, train=False)
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
    print("dead trees predicted")
    #TODO confirm this is the correct axis
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
    
    sns.set_theme(style="whitegrid")
    
    g = sns.countplot(
        data=results,
        x="plantStatus", hue="Dead", palette="dark")
    fig = g.get_figure()
    fig.savefig("figures/plantStatus.png")
    
    #Plot trees that are incorrect
    
    return results

if __name__ == "__main__":
    results = run(
        checkpoint_path="/orange/idtrees-collab/DeepTreeAttention/Dead/cef3e91d8a9c4e848d85d333233b3c7f.pl",
        image_dir="/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB",
        savedir="figures/",
        field_path="data/neon_vst_data_2021.csv",
        num_workers=10,
    debug=False)
    results.to_csv("figures/results.csv")
