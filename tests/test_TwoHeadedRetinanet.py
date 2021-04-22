#test two class headed output

from TwoHeadedRetinanet import TwoHeadedRetinanet
from deepforest import main
from deepforest import get_data
from deepforest import preprocess
from deepforest.visualize import format_boxes, plot_prediction_dataframe
import os
from skimage import io
import torch
import numpy as np

def test_TwoHeadedRetinanet_predict():
    original_model = main.deepforest()
    original_model.use_release()
    
    m = TwoHeadedRetinanet(trained_model=original_model.model)
    m.eval()
    
    image_path = get_data("OSBS_029.png")
    image = io.imread(image_path)
    x = preprocess.preprocess_image(image)

    assert m.head.classification_head_task1.num_classes==1
    assert m.head.classification_head_task2.num_classes==2    
    
    #set the nms thresh and score thresh as comparable to the original model
    m.nms_thresh = original_model.model.nms_thresh
    m.score_thresh = original_model.model.score_thresh
    
    prediction = m(x)    
    assert list(prediction[0].keys()) == ["boxes","scores","labels","scores_task2","labels_task2"]
    
    #Boxes in task1 should be identical to original model 
    original_model.model.eval()
    original_prediction = original_model.model(x)
    assert torch.equal(prediction[0]["boxes"],original_prediction[0]["boxes"])
    assert torch.equal(prediction[0]["scores"],original_prediction[0]["scores"])
    
    assert not torch.equal(prediction[0]["scores_task2"],original_prediction[0]["scores"])
    
    #Correct number of classes
    assert all([x.numpy() in np.arange(m.head.classification_head_task2.num_classes) for x in prediction[0]["labels_task2"]])
    
    task1_df = format_boxes(prediction[0])
    task1_df["image_path"] = os.path.basename(image_path)
    
    #View predictions
    plot_prediction_dataframe(df= task1_df, root_dir=os.path.dirname(image_path), show=True)
    
    
def test_TwoHeadedRetinanet_train():
    original_model = main.deepforest()
    original_model.use_release()
    
    m = TwoHeadedRetinanet(trained_model=original_model.model)
    csv_file = get_data("OSBS_029.csv")
    
    original_model.config["workers"] = 0
    ds = original_model.load_dataset(csv_file=csv_file, root_dir=os.path.dirname(csv_file), augment=False, shuffle=True, batch_size=1)
    
    #TODO, the targets are current 0/1 from dead alive, it needs to pass target labels for EACH of the tasks.
    
    batch = next(iter(ds))
    image_path, image, targets = batch
    forward_pass = m(image, targets)    
