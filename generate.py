#Generate data for model training
#DeepForest 19 site model
import pandas as pd
import glob
import os
import numpy as np

from deepforest import utilities
from deepforest import preprocess

#dask integration
from dask_utility import start_dask_cluster
from dask.distributed import wait

def generate_pretraining(DEBUG, BASE_PATH, DATA_PATH, BENCHMARK_PATH,dask_client=None, allow_empty=False):
    
    #Remove previous files if needed
    previous_files = ["pretraining/pretraining_annotations.csv","pretraining/crops/pretraining.csv","pretraining/crops/classes.csv"]
    for f in previous_files:
        full_path = os.path.join(BASE_PATH,f)
        if os.path.exists(full_path):
            os.remove(full_path)
            
    #Collect pretraining annotations
    data_paths = glob.glob(BASE_PATH + "pretraining/*.csv")
    dataframes = (pd.read_csv(f, index_col=None) for f in data_paths)
    annotations = pd.concat(dataframes, ignore_index=True)      
    
    #first column is the image path, set float to int for cells
    annotations.head()
    annotations.rename(columns={"plot_name":"image_path"},inplace=True)
        
    annotations.xmin = annotations.xmin.astype(int)
    annotations.ymin = annotations.ymin.astype(int)
    annotations.xmax = annotations.xmax.astype(int)
    annotations.ymax = annotations.ymax.astype(int)
    
    #Load benchmark annotations and eliminate any pretraining tiles that come from same geographic area
    benchmark = pd.read_csv(BENCHMARK_PATH + "evaluation/RGB/benchmark_locations.csv")
    
    #Get geoindex of each annotations
    annotations["geo_index"] = annotations.image_path.str.extract("_(\\d+_\\d+)_image")
    
    print("Removing {} of {} tiles from pretraining to give 1km buffer to evaluation".format(
        sum(annotations.geo_index.isin(benchmark["geo_index"])), 
        annotations.shape[0]))
    
    annotations = annotations[~annotations.geo_index.isin(benchmark["geo_index"])]
    
    #drop column
    annotations.drop(columns=['geo_index'], inplace=True)
    
    #HOTFIX!, the current detection paths are not relative.
    annotations["image_path"] =  annotations["image_path"].apply(lambda x: os.path.basename(x))    
    annotations.to_csv(BASE_PATH + "pretraining/pretraining_annotations.csv", index=False)
    
    #Find training tiles and crop into overlapping windows for detection
    

    
    #Find all tifs available
    image_index = annotations.image_path.unique()        
    all_tifs = glob.glob(DATA_PATH + "**/*.tif", recursive=True)
    tif_basename = [os.path.basename(x) for x in all_tifs]
    selected_indices = [tif_basename.index(x) for x in image_index if x in tif_basename ]
    raster_list = np.array(all_tifs)[selected_indices]
        
    print("There are {} tiles to process".format(len(raster_list)))
    
    cropped_annotations = [ ]
    
    if dask_client:
        futures = dask_client.map(preprocess.split_raster,
                                  raster_list,
                                  annotations_file=BASE_PATH + "pretraining/pretraining_annotations.csv",
                                  base_dir=BASE_PATH + "pretraining/crops/",
                                  patch_size=400,
                                  patch_overlap=0.05,
                                  allow_empty=allow_empty)
        
        wait(futures)
        
        for future in futures:
                try:
                    local_annotations = future.result()
                    cropped_annotations.append(local_annotations)
                except Exception as e:
                    print("future {} failed with {}".format(future, e))
    else:
        for raster in raster_list:
            annotations_df= preprocess.split_raster(path_to_raster=raster,
                                             annotations_file=BASE_PATH + "pretraining/pretraining_annotations.csv",
                                             base_dir=BASE_PATH + "pretraining/crops/",
                                             patch_size=400,
                                             patch_overlap=0.05)
            cropped_annotations.append(annotations_df)
            #Write individual in case we want to generate tfrecords            
    
    ##Gather annotation files into a single file
    annotations = pd.concat(cropped_annotations, ignore_index=True)      
    
    #if DEBUG:
       #annotations = annotations[0:100]
    
    #clean make sure non NA boxes have area greater than 2 pixels
    blank_images = annotations[annotations.xmin==""]
    non_blank_images = annotations[~(annotations.xmin=="")]
    non_blank_images = non_blank_images[(non_blank_images.xmin < non_blank_images.xmax -2)]
    non_blank_images = non_blank_images[(non_blank_images.ymin < non_blank_images.ymax -2)]
    combined_annotations = pd.concat([blank_images, non_blank_images])
    
    #sort by image_path
    combined_annotations.sort_values(by="image_path", axis=0, ascending=True, 
                                    inplace=True, 
                                    kind="quicksort", 
                                    na_position="last")
    
    combined_annotations.to_csv(BASE_PATH + "pretraining/crops/pretraining.csv", index=False, header=None)
    
def generate_training(DEBUG, BASE_PATH, dask_client=None, allow_empty=False):
    
    #Remove previous files if needed
    previous_files = ["hand_annotations/crops/hand_annotations.csv", "hand_annotations/hand_annotations.csv","hand_annotations/crops/classes.csv"]
    for f in previous_files:
        full_path = os.path.join(BASE_PATH,f)
        if os.path.exists(full_path):
            os.remove(full_path)
            
    ## Hand annotations ##
    #convert hand annotations from xml into retinanet format
    xmls = glob.glob(BASE_PATH + "hand_annotations/*.xml")
    annotation_list = []
    for xml in xmls:
        annotation = utilities.xml_to_annotations(xml)
        annotation_list.append(annotation)
    
    #Collect hand annotations
    annotations = pd.concat(annotation_list, ignore_index=True)      
    
    #force dtype
    annotations.xmin = annotations.xmin.astype(int)
    annotations.ymin = annotations.ymin.astype(int)
    annotations.xmax = annotations.xmax.astype(int)
    annotations.ymax = annotations.ymax.astype(int)
    
    annotations.to_csv(BASE_PATH + "hand_annotations/hand_annotations.csv",index=False)
    
    #Collect hand annotation tiles
    xmls = glob.glob(BASE_PATH+"hand_annotations/*.xml")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in xmls] 
    raster_list = [ BASE_PATH + "hand_annotations/" + x + ".tif" for x in xmls] 
    
    if DEBUG:
        raster_list=[raster_list[0]]
    
    cropped_annotations = [ ]
    
    if dask_client:
        futures = dask_client.map(preprocess.split_raster,
                                  raster_list,
                                  annotations_file=BASE_PATH + "hand_annotations/hand_annotations.csv",
                                  base_dir=BASE_PATH + "hand_annotations/crops/",
                                  patch_size=400,
                                  patch_overlap=0.05,
                                  allow_empty=allow_empty)
        
        wait(futures)
    
        for future in futures:
            try:
                local_annotations = future.result()
                cropped_annotations.append(local_annotations)
            except Exception as e:
                print("future {} failed with {}".format(future, e))        
        print("hand annotation generation complete")

    else:
        for raster in raster_list:
            annotations_df= preprocess.split_raster(path_to_raster=raster,
                                             annotations_file=BASE_PATH + "hand_annotations/hand_annotations.csv",
                                             base_dir=BASE_PATH + "hand_annotations/crops/",
                                             patch_size=400,
                                             patch_overlap=0.05)
            cropped_annotations.append(annotations_df)
    
    ##Gather annotation files into a single file
    annotations = pd.concat(cropped_annotations, ignore_index=True)   
    
    #Ensure column order
    annotations.to_csv(BASE_PATH + "hand_annotations/crops/hand_annotations.csv",index=False, header=None)
    print(annotations.head())

## Benchmark data for validation ##
def generate_benchmark(BENCHMARK_PATH):
    tifs = glob.glob(BENCHMARK_PATH + "evaluation/RGB/*.tif")
    xmls = [os.path.splitext(os.path.basename(x))[0] for x in tifs] 
    xmls = [os.path.join(BENCHMARK_PATH, "annotations", x) + ".xml" for x in xmls] 
    
    #Load and format xmls, not every RGB image has an annotation
    annotation_list = []   
    for xml_path in xmls:
        try:
            annotation = utilities.xml_to_annotations(xml_path)
        except:
            pass
        annotation_list.append(annotation)
    benchmark_annotations = pd.concat(annotation_list, ignore_index=True)      
    
    #save evaluation annotations
    fname = os.path.join(BENCHMARK_PATH + "evaluation/RGB/benchmark_annotations.csv")
    benchmark_annotations.to_csv(fname, index=False, header=None)
    
if __name__=="__main__":
    #Local debug. If False, paths on UF hypergator supercomputing cluster
    DEBUG = False
  
    if DEBUG:
        BASE_PATH = "/Users/ben/Documents/DeepForest_Model/"
        DATA_PATH = "/Users/ben/Documents/DeepForest_Model/"
        BENCHMARK_PATH = "/Users/ben/Documents/NeonTreeEvaluation/"
        dask_client = None
    else:
        BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"
        BENCHMARK_PATH = "/home/b.weinstein/NeonTreeEvaluation/"
        DATA_PATH = "/orange/ewhite/NeonData/"        
        dask_client = start_dask_cluster(number_of_workers=60, mem_size="11GB")
    
    #Run Benchmark
    #generate_benchmark(BENCHMARK_PATH)
        
    #Run pretraining
    generate_pretraining(DEBUG, BASE_PATH, DATA_PATH, BENCHMARK_PATH, dask_client, allow_empty=False)
    
    #Run Training
    #generate_training(DEBUG, BASE_PATH, dask_client, allow_empty=True)
    

    
