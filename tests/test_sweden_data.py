#test sweden data
import sweden_data
from deepforest.visualize import plot_predictions
from skimage import io

def test_xml_to_annotations():
    df = sweden_data.xml_to_annotations("tests/data/B10_0046.xml")
    assert all([x in df.columns for x in ["xmin","ymin","xmax","ymax","label"]])
    assert df.shape[0] == 60
    
    image = io.imread("tests/data/B10_0046.JPG")
    
    #plot 
    plot_predictions(image=image, df=df)