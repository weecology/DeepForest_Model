#Alive Dead Model, optionally building from release tree crown model
import comet_ml
import glob
import time
import random
from datetime import datetime
import os
from deepforest import main
from pytorch_lightning.loggers import CometLogger

def match_state_dict(state_dict_a, state_dict_b):
    """ Filters state_dict_b to contain only states that are present in state_dict_a. Contributed by hgaiser
    https://github.com/pytorch/pytorch/pull/39144#issuecomment-784560497

    state_dict_a: Dict[str, torch.Tensor],
        state_dict_b: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    Matching happens according to two criteria:
        - Is the key present in state_dict_a?
        - Does the state with the same key in state_dict_a have the same shape?

    Returns
        (matched_state_dict, unmatched_state_dict)

        States in matched_state_dict contains states from state_dict_b that are also
        in state_dict_a and unmatched_state_dict contains states that have no
        corresponding state in state_dict_a.

    	In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
    """
    matched_state_dict = {
            key: state
                for (key, state) in state_dict_b.items()
                if key in state_dict_a and state.shape == state_dict_a[key].shape
        }
    unmatched_state_dict = {
            key: state
                for (key, state) in state_dict_b.items()
                if key not in matched_state_dict
        }
    return matched_state_dict, unmatched_state_dict

def train(train_path, test_path, pretrained=False, image_dir = "/orange/idtrees-collab/NeonTreeEvaluation/evaluation/RGB/", debug=False, savedir="/orange/idtrees-collab/DeepTreeAttention/Dead/"):
    
    comet_logger = CometLogger(api_key="ypQZhYfs3nSyKzOfz13iuJpj2",
                                  project_name="deepforest-pytorch", workspace="bw4sz")
    comet_logger.experiment.add_tag("DeadAlive")
    
    #add small sleep for SLURM jobs
    time.sleep(random.randint(0,10))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comet_logger.experiment.log_parameter("timestamp", timestamp)
    savedir = "{}/{}".format(savedir,timestamp)
    
    try:
        os.mkdir(savedir)
    except:
        pass
    
    #Get release state dict
    release_model = main.deepforest()
    release_model.use_release()
    
    #Two class tree model
    m = main.deepforest(num_classes=2, label_dict={"Alive":0,"Dead":1})
    
    #filter matching 
    matched_state_dict = match_state_dict(state_dict_a=m.state_dict(), state_dict_b=release_model.state_dict())
    
    m.load_state_dict(matched_state_dict[0],strict=False)
    
    m.config["train"]["csv_file"] = train_path
    m.config["train"]["root_dir"] = image_dir
    m.config["validation"]["csv_file"] = test_path
    m.config["validation"]["root_dir"] = image_dir
    
    if debug:
        m.config["train"]["fast_dev_run"] = True
        m.config["gpus"] = None
        m.config["workers"] = 0
    
    m.create_trainer()
    
    m.trainer.fit(m)
    
    result_dict = m.evaluate(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"])
    
    comet_logger.experiment.log_metric("test_box_precision",result_dict["box_precision"])
    comet_logger.experiment.log_metric("test_box_recall",result_dict["box_recall"])
    
    result_dict["class_recall"].to_csv("{}/class_recall.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/class_recall.csv".format(savedir))
    
    result_dict["results"].to_csv("{}/results.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/results.csv".format(savedir))
    
    boxes = m.predict_file(csv_file=m.config["validation"]["csv_file"], root_dir=m.config["validation"]["root_dir"], savedir=savedir)
    images = glob.glob("{}/*.png".format(savedir))
    random.shuffle(images)
    for img in images[:20]:
        comet_logger.experiment.log_image(img)
    boxes.to_csv("{}/benchmark_predictions.csv".format(savedir))
    comet_logger.experiment.log_asset("{}/benchmark_predictions.csv".format(savedir))
    
    m.save_model("{}/hand_annotated.pl".format(savedir))
    comet_logger.experiment.log_parameter("saved model", "{}/hand_annotated.pl".format(savedir))

             
if __name__ == "__main__":
    train(train_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv",
          test_path="/orange/idtrees-collab/DeepTreeAttention/data/dead_train.csv")