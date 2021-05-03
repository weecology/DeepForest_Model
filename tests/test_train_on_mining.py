#Test training on the hard mined data
import sys
import os
sys.path.insert(0, "Dead")
from Dead import train_on_mining

os.environ['KMP_DUPLICATE_LIB_OK']='True'
def test_run(tmpdir):
    train_on_mining.run(
        checkpoint="snapshots/alive_dead.pl",
        annotation_dir="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/annotations/",
        csv_dir="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/",
        image_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB",
        savedir=tmpdir,
        fast_dev_run=True,
        gpus=None,
        num_workers=0)
