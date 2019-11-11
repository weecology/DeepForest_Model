## Prediction of Trees from local model
import cv2
import matplotlib.pyplot as plt  
import pandas as pd
import glob
import os
from deepforest import deepforest

# Benchmark data for validation
PATH_TO_BENCHMARK_DATA = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/*.tif"
files = glob.glob(PATH_TO_BENCHMARK_DATA)

#Load model
prediction_model = deepforest.deepforest(weights="snapshots/final_weights.h5")

boxes_output = []
for f in files:
    print(f)
    #predict plot image
    boxes = prediction_model.predict_image(f, show=False, return_plot=False)
    box_df = pd.DataFrame(boxes)
    
    #plot name
    plot_name = os.path.splitext(os.path.basename(f))[0]
    box_df["plot_name"] = plot_name
    boxes_output.append(box_df)
    
boxes_output = pd.concat(boxes_output)

#name columns and add to submissions folder
boxes_output.columns = ["xmin","ymin","xmax","ymax","plot_name"]
boxes_output = boxes_output.reindex(columns= ["plot_name","xmin","ymin","xmax","ymax"])    
boxes_output.to_csv("../submissions/Weinstein_unpublished.csv",index=False)

