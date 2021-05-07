#test predict
import predict
import glob

def test_predict(tmpdir):
    predict.run(checkpoint_path="snapshots/alive_dead.pl", image_glob="tests/data/*.tif", shape_dir="tests/data/", savedir=tmpdir, num_workers=0)
    assert len(glob.glob("{}/*.shp".format(tmpdir))) == 1