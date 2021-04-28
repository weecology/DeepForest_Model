#test mining
import sys
sys.path.insert(0, "Dead")
import mining
import torch
import geopandas as gpd
from vanilla import AliveDeadVanilla  

def test_mine_dead(tmpdir):
    
    m = AliveDeadVanilla()
    torch.save(m.model.state_dict(),"{}/model.pt".format(tmpdir))
    model_path = "{}/model.pt".format(tmpdir)
    
    image_path = "Dead/data/SOAP_052_2018.tif"
    shp = gpd.read_file("Dead/data/SOAP_052_2018_327863759.shp")
    shp["shp_path"] = "Dead/data/SOAP_052_2018_327863759.shp"
    shp = shp.rename(columns={"box_utm_le":"left","box_utm_ri":"right","box_utm_to":"top","box_utm_bo":"bottom"})
    shp.to_file("{}/example.shp".format(tmpdir))
    #csv file format is left,bottom,right,top,score,label,height,area,shp_path,geo_index,Year,Site
    
    mining.mine_dead(shp="{}/example.shp".format(tmpdir), image_path=image_path, model_path=model_path, savedir=tmpdir)