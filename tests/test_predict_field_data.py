#Test predict_field_data
import sys
import os
sys.path.insert(0, "Dead")
from Dead import predict_field_data

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_run(tmpdir):
    results = predict_field_data.run(
        checkpoint_path="snapshots/alive_dead.pl",
        image_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB",
        savedir=tmpdir,
        field_path="Dead/data/neon_vst_data_2021.csv",
        num_workers=0,
    debug=False)
    results.to_csv("Dead/figures/results.csv")
    