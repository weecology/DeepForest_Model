#test alive dead
import alive_dead
import os

def test_train(tmpdir):
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    alive_dead.train(train_path="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_train.csv",
                     test_path="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/dead_test.csv",
                     image_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB",
                     debug=True)