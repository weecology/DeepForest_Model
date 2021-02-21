from src import crops
import glob
from deepforest.preprocess import split_raster
from deepforest import utilities
import rasterio


def test_shapefile_to_annotations(tmpdir):
    df = utilities.shapefile_to_annotations(shapefile="/Users/benweinstein/Downloads/temp_training/2019_OSBS_5_410000_3282000_image_crop.shp", rgb="/Users/benweinstein/Downloads/temp_training/2019_OSBS_5_410000_3282000_image_crop.tif")
    df.to_csv("{}/annotations.csv".format(tmpdir))
    split_df = split_raster("/Users/benweinstein/Downloads/temp_training/2019_OSBS_5_410000_3282000_image_crop.tif",
                            annotations_file="{}/annotations.csv".format(tmpdir),base_dir=tmpdir, patch_size=400,
                            patch_overlap=0.05,
                            allow_empty=False)
    
    created_crops = glob.glob("{}/*".format(tmpdir))
    b = rasterio.open("{}/2019_OSBS_5_410000_3282000_image_crop_0.png".format(tmpdir))
    annotations_per_crop = split_df.groupby("image_path").size()
    assert all(annotations_per_crop < 200)
    
    