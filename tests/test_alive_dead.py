#test alive dead
import alive_dead
from deepforest import main
from deepforest import get_data
import pandas as pd
import os
import torch

def assert_state_dict_equal(model_1, model_2):
    """Assert that two pytorch model state dicts are identical
    from https://discuss.pytorch.org/t/two-models-with-same-weights-different-results/8918/7
    Args:
        model_1: a state_dict object from a model
        model_2: a state_dict object from a 2nd model
    Returns:
        None: assertion that models are the same
    """
    models_differ = 0    
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    assert models_differ == 0
        
def test_train(tmpdir):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    m = alive_dead.train(train_path="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_train.csv",
                     test_path="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_test.csv",
                     image_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB",
                     savedir=tmpdir,
                     debug=True)
    
    #TODO Assert that the 2nd classification weights are changed, but not the regression weights when compared to the release model.
    original_model = main.deepforest()
    original_model.use_release()
    
    #Backbones are the same
    assert_state_dict_equal(model_1 = original_model.model.backbone.state_dict(), model_2=m.model.backbone.state_dict())
    assert_state_dict_equal(model_1 = original_model.model.head.classification_head.state_dict(), model_2=m.model.head.classification_head_task1.state_dict())
    assert_state_dict_equal(model_1 = original_model.model.head.regression_head.state_dict(), model_2=m.model.head.regression_head.state_dict())
    
    #models should produce identical box predictions
    img_path = get_data("OSBS_029.png")
    original_boxes = original_model.predict_image(path=img_path)
    final_boxes = m.predict_image(path=img_path)
    pd.testing.assert_frame_equal(original_boxes.drop(columns="label"), final_boxes.drop(columns="label"))
    
    #But the new classification head should have updated weights
    
def test_trained_evaluation():
    """download a sample trained model to look at eval code"""
    
    trained_model = alive_dead.main.deepforest.load_from_checkpoint("snapshots/alive_dead.pl")
    trained_model.label_dict()