"""
Generate tfrecords by creating the anchor boxes for each retinanet batch
"""
import numpy as np
import pandas as pd
from math import ceil

from deepforest.utilities import read_config
from deepforest import tfrecords
from deepforest import utilities

from dask_utility import start_dask_cluster
from dask.distributed import wait

def generate_pretraining(DEBUG, BASE_PATH, FILEPATH, SIZE,config,dask_client):
    annotations_file = BASE_PATH + "pretraining/crops/pretraining.csv"
    class_file = utilities.create_classes(annotations_file)

    #Split annotations file into chunks

    if DEBUG:
        tfrecords.create_tfrecords(annotations_file=annotations_file,
                                   class_file=class_file,
                                   image_min_side=config["image-min-side"],
                                   backbone_model=config["backbone"],
                                   size=SIZE,
                                   savedir=FILEPATH + "pretraining/tfrecords/")
    else:
        #Collect annotation files for each tile
        annotations_file= BASE_PATH + "pretraining/crops/pretraining.csv"
        df = pd.read_csv(annotations_file, names=["image_path","xmin","ymin","xmax","ymax","label"])

        #enforce dtype if NAs are present
        df.xmin = df.xmin.astype(pd.Int64Dtype())
        df.ymin = df.ymin.astype(pd.Int64Dtype())
        df.xmax = df.xmax.astype(pd.Int64Dtype())
        df.ymax = df.ymax.astype(pd.Int64Dtype())

        #Randomize rows
        df = df.sample(frac=1)

        #Show head
        df.head()

        #split pandas frame into chunks
        images = df.image_path.unique()
        indices = np.arange(len(images))

        #Number of images per dask worker
        size = 10000
        chunk_list = [ ]

        #Split dataframe into chunks of images and write to file
        for i in range(ceil(len(indices) / size)):
            image_indices = indices[i * size:(i * size) + size]
            selected_images = images[image_indices]
            split_frame = df[df.image_path.isin(selected_images)]
            filename = BASE_PATH + "pretraining/crops/pretraining_{}.csv".format(i)
            split_frame.to_csv(filename, header=False,index=False)
            chunk_list.append(filename)

        print(" Created {} files to create tfrecords".format(len(chunk_list)))

        #Apply create tfrecords to each tile
        futures = dask_client.map(
            tfrecords.create_tfrecords,
            chunk_list,
            class_file=class_file,
            image_min_side=config["image-min-side"],
            backbone_model=config["backbone"],
            size=SIZE,
            savedir=FILEPATH + "pretraining/tfrecords/")

        wait(futures)
        for future in futures:
            try:
                local_annotations = future.result()
            except Exception as e:
                print("future {} failed with {}".format(future, e))

def generate_hand_annotations(DEBUG, BASE_PATH, FILEPATH, SIZE, config, dask_client):
    
    #Generate tfrecords
    dirname = "hand_annotations/"

    annotations_file = BASE_PATH + dirname + "crops/hand_annotations.csv"

    class_file = utilities.create_classes(annotations_file)

    if DEBUG:
        tfrecords.create_tfrecords(annotations_file=annotations_file,
                                   class_file=class_file,
                                   image_min_side=config["image-min-side"],
                                   backbone_model=config["backbone"],
                                   size=SIZE,
                                   savedir=FILEPATH + dirname + "tfrecords/")
    else:

        #Collect annotation files for each tile
        annotations_file= BASE_PATH + dirname + "crops/hand_annotations.csv"
        df = pd.read_csv(annotations_file, names=["image_path","xmin","ymin","xmax","ymax","label"])

        #enforce dtype, as there might be errors
        df.xmin = df.xmin.astype(pd.Int64Dtype())
        df.ymin = df.ymin.astype(pd.Int64Dtype())
        df.xmax = df.xmax.astype(pd.Int64Dtype())
        df.ymax = df.ymax.astype(pd.Int64Dtype())

        #Randomize rows
        df = df.sample(frac=1)

        #split pandas frame into chunks
        images = df.image_path.unique()
        indices = np.arange(len(images))
        size = 500

        chunk_list = [ ]

        #Split dataframe into chunks of images and write to file
        for i in range(ceil(len(indices) / size)):
            image_indices = indices[i * size:(i * size) + size]
            selected_images = images[image_indices]
            split_frame = df[df.image_path.isin(selected_images)]
            filename = BASE_PATH + dirname + "crops/hand_annotations{}.csv".format(i)
            split_frame.to_csv(filename, header=False,index=False)
            chunk_list.append(filename)

        print(" Created {} files to create tfrecords".format(len(chunk_list)))

        #Apply create tfrecords to each
        futures = dask_client.map(
            tfrecords.create_tfrecords,
            chunk_list,
            class_file=class_file,
            image_min_side=config["image-min-side"],
            backbone_model=config["backbone"],
            size=SIZE,
            savedir=FILEPATH + dirname + "tfrecords/")

        wait(futures)
        for future in futures:
            try:
                local_annotations = future.result()
            except Exception as e:
                print("future {} failed with {}".format(future, e))

def generate_benchmark(DEBUG, BENCHMARK_PATH, FILEPATH, SIZE, config, dask_client):
    #Generate tfrecords

    annotations_file = BENCHMARK_PATH + "evaluation/RGB/benchmark_annotations.csv"

    class_file = utilities.create_classes(annotations_file)

    if DEBUG:
        tfrecords.create_tfrecords(annotations_file=annotations_file,
                                   class_file=class_file,
                                   image_min_side=config["image-min-side"],
                                   backbone_model=config["backbone"],
                                   size=SIZE,
                                   savedir=FILEPATH + "evaluation/RGB/tfrecords/")
    else:

        #Collect annotation files for each tile
        df = pd.read_csv(annotations_file, names=["image_path","xmin","ymin","xmax","ymax","label"])

        #Randomize rows
        df = df.sample(frac=1)

        #enforce dtype, as there might be errors
        df.xmin = df.xmin.astype(pd.Int64Dtype())
        df.ymin = df.ymin.astype(pd.Int64Dtype())
        df.xmax = df.xmax.astype(pd.Int64Dtype())
        df.ymax = df.ymax.astype(pd.Int64Dtype())

        df.head()

        #split pandas frame into chunks
        images = df.image_path.unique()
        indices = np.arange(len(images))
        size = 500

        chunk_list = [ ]

        #Split dataframe into chunks of images and write to file
        for i in range(ceil(len(indices) / size)):
            image_indices = indices[i * size:(i * size) + size]
            selected_images = images[image_indices]
            split_frame = df[df.image_path.isin(selected_images)]
            filename = BENCHMARK_PATH + "evaluation/RGB/benchmark_annotations_{}.csv".format(i)
            split_frame.to_csv(filename, header=False,index=False)
            chunk_list.append(filename)

        print(" Created {} files to create tfrecords".format(len(chunk_list)))

        #Apply create tfrecords to each
        futures = dask_client.map(
            tfrecords.create_tfrecords,
            chunk_list,
            class_file=class_file,
            image_min_side=config["image-min-side"],
            backbone_model=config["backbone"],
            size=SIZE,
            savedir=FILEPATH + "evaluation/RGB/tfrecords/")

        wait(futures)
        for future in futures:
            try:
                local_annotations = future.result()
            except Exception as e:
                print("future {} failed with {}".format(future, e))

if __name__=="__main__":

    #Generate anchor objects for each image and wrap in tfrecords
    DEBUG = False

    #Number of images per tfrecord
    SIZE = 50

    #Set paths
    if DEBUG:
        BASE_PATH = "/Users/ben/Documents/DeepForest_Model/"
        FILEPATH = "/Users/ben/Documents/DeepForest_Model/"
        BENCHMARK_PATH = "/Users/ben/Documents/NeonTreeEvaluation/"
        dask_client = None
    else:
        BASE_PATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"
        FILEPATH = "/orange/ewhite/b.weinstein/NeonTreeEvaluation/"
        BENCHMARK_PATH = "/home/b.weinstein/NeonTreeEvaluation/"
        dask_client = start_dask_cluster(number_of_workers=15, mem_size="5GB")

    #Read config
    config = read_config("deepforest_config.yml")

    #generate_hand_annotations(DEBUG, BASE_PATH, FILEPATH, SIZE, config, dask_client)
    #generate_pretraining(DEBUG, BASE_PATH, FILEPATH, SIZE, config, dask_client)
    generate_benchmark(DEBUG, BENCHMARK_PATH, BENCHMARK_PATH, SIZE, config, dask_client)
