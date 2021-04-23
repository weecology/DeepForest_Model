#Predict 2nd auxillary task for evaluate
import pandas as pd
from deepforest import preprocess
from deepforest import visualize
from deepforest.predict import across_class_nms
from skimage import io
import os

def format_boxes(prediction, scores=True):
    """Format a retinanet prediction into a pandas dataframe for a single image
       Args:
           prediction: a dictionary with keys 'boxes' and 'labels' coming from a retinanet
           scores: Whether boxes come with scores, during prediction, or without scores, as in during training.
        Returns:
           df: a pandas dataframe
    """

    df = pd.DataFrame(prediction["boxes"].cpu().detach().numpy(),
                      columns=["xmin", "ymin", "xmax", "ymax"])
    df["label"] = prediction["labels_task2"].cpu().detach().numpy()

    #TODO return both sets of scores?
    if scores:
        df["score"] = prediction["scores"].cpu().detach().numpy()

    return df

def predict_file(model, csv_file, root_dir, savedir, device, iou_threshold=0.1):
    """Create a dataset and predict entire annotation file

    Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
    Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

    Args:
        csv_file: path to csv file
        root_dir: directory of images. If none, uses "image_dir" in config
        savedir: Optional. Directory to save image plots.
        device: pytorch device of 'cuda' or 'cpu' for gpu prediction. Set internally.
    Returns:
        df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
    """

    input_csv = pd.read_csv(csv_file)

    # Just predict each image once.
    images = input_csv["image_path"].unique()

    prediction_list = []
    for path in images:
        image = io.imread("{}/{}".format(root_dir, path))

        image = preprocess.preprocess_image(image)

        # Just predict the images, even though we have the annotations
        if not device.type == "cpu":
            image = image.to(device)

        prediction = model(image)
        
        prediction = format_boxes(prediction[0])
        prediction = across_class_nms(prediction, iou_threshold = iou_threshold)
        
        prediction["image_path"] = path
        prediction_list.append(prediction)

        if savedir:
            #if on GPU, bring back to cpu for plotting
            # Just predict the images, even though we have the annotations
            if not device.type == "cpu":
                image = image.to("cpu")
                
            image = image.squeeze(0).permute(1, 2, 0)
            plot, ax = visualize.plot_predictions(image, prediction)
            annotations = input_csv[input_csv.image_path == path]
            plot = visualize.add_annotations(plot, ax, annotations)
            plot.savefig("{}/{}.png".format(savedir, os.path.splitext(path)[0]),dpi=300)

    df = pd.concat(prediction_list, ignore_index=True)

    return df