#Test vanilla
from Dead import train_vanilla
import pytest

@pytest.mark.parametrize("alive_weight",[None, 10])
def test_AliveDeadVanilla(alive_weight):
    train_vanilla.run(csv_dir="/Users/benweinstein/Dropbox/Weecology/TreeDetectionZooniverse/",
                root_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB",
                alive_weight=alive_weight, gpus=None, batch_size=1, num_workers=0, fast_dev_run=True)