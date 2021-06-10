import xmltodict
import numpy as np
import glob
import os
import pandas as pd

def xml_to_annotations(xml_path):
    """
    Load annotations from xml format (e.g. RectLabel editor) and convert
    them into retinanet annotations format.
    Args:
        xml_path (str): Path to the annotations xml, formatted by RectLabel
    Returns:
        Annotations (pandas dataframe): in the
            format -> path-to-image.png,x1,y1,x2,y2,class_name
    """
    # parse
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())

    # grab xml objects
    try:
        tile_xml = doc["annotation"]["object"]
    except Exception as e:
        raise Exception("error {} for path {} with doc annotation{}".format(
            e, xml_path, doc["annotation"]))

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    label = []

    if isinstance(tile_xml, list):
        # Construct frame if multiple trees
        for tree in tile_xml:
            xmin.append(tree["bndbox"]["xmin"])
            xmax.append(tree["bndbox"]["xmax"])
            ymin.append(tree["bndbox"]["ymin"])
            ymax.append(tree["bndbox"]["ymax"])
            label.append(tree['tree'])
    else:
        xmin.append(tile_xml["bndbox"]["xmin"])
        xmax.append(tile_xml["bndbox"]["xmax"])
        ymin.append(tile_xml["bndbox"]["ymin"])
        ymax.append(tile_xml["bndbox"]["ymax"])
        label.append(tile_xml['tree'])

    rgb_name = os.path.basename(doc["annotation"]["filename"])

    # set dtypes, check for floats and round
    xmin = [int(np.round(float(x))) for x in xmin]
    xmax = [int(np.round(float(x)))for x in xmax]
    ymin = [int(np.round(float(x))) for x in ymin]
    ymax = [int(np.round(float(x))) for x in ymax]

    annotations = pd.DataFrame({
        "image_path": rgb_name,
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "label": label
    })
    return (annotations)

def prepare_data(paths):
    """Loop through the xml data, create a train/test split csv files and create"""    
    results = []
    for x in paths:
        try:
            df = xml_to_annotations(x)
        except Exception as e:
            print("{} failed with {}".format(x, e))
        
        results.append(df)
    
    results = pd.concat(results)
    results = results.reset_index(drop=True)
    
    return results
    
    
if __name__ == "__main__":
    paths = glob.glob("/orange/ewhite/b.weinstein/Radogoshi_Sweden/*/Annotations/*.xml", recursive=True)
    results = prepare_data(paths)
    results.to_csv("/orange/ewhite/b.weinstein/Radogoshi_Sweden/annotations.csv")