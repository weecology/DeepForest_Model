#Create a callback to log images during training
from deepforest.callbacks import Callback
import numpy as np

def comet_callbacks(Callback):
    def __init__(self, model, epoch, dataset, experiment, n):
        """
        Predict a sample of n images to log and view
        Args:
            model: a pytorch model object
            epoch: current epoch integer
            dataset: a pytorch dataset iterator
            experiment: active comet_ml experiment class
            n: number of images to plot
        Returns:
            None: images are written as a side effect
        """
        self.experiment = experiment
    
    def on_epoch_end(self):
        for i in range(np.arange(self.n)):
            batch = next(self.dataset)
            predictions = self.model(batch)
            
        
        #yield 
        