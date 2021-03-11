"""
Dataset model
https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:
boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
labels (Int64Tensor[N]): the class label for each ground-truth box
"""

import glob
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from deepforest import transforms as T

idx_to_label = {
    "Tree": 0
}

class TreeDirectory(Dataset):
    def __init__(self, csv_dir, root_dir, transforms):
        """
        An out of memory dataset device to load annotations from individual .csv files for each image.
        Args:
            csv_dir (string): Path to the directory of csv files with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = glob.glob("{}/*.csv".format(csv_dir))
        self.root_dir = root_dir
        self.transform = transforms
        #create a dictionary of filenames and crops
        self.image_dict = {}
        counter = 0
        for x in self.files:
            annotations = pd.read_csv(x)
            for image_path in annotations.image_path.unique():
                self.image_dict[counter] = {"file":x, "image_path": image_path}
                counter = counter + 1
                

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, idx):
        file_dict = self.image_dict[idx]
        csv_file=file_dict["file"]
        tile_annotations = pd.read_csv(csv_file)
        image_annotations = tile_annotations[tile_annotations.image_path == file_dict["image_path"]]
        img_name = os.path.join(self.root_dir, image_annotations.image_path.unique()[0])
        
        image = io.imread(img_name)
        image = image/255   
        
        targets = {}
        targets["boxes"] = image_annotations[["xmin", "ymin", "xmax",
                                   "ymax"]].values.astype(float)
        
        #Labels need to be encoded? 0 or 1 indexed?, ALl tree for the moment.
        targets["labels"] = image_annotations.label.apply(lambda x: idx_to_label[x]).values.astype(int)

        if self.transform:
            image, targets = self.transform(image, targets)

        return img_name, image, targets    