#Test tfrecords to make sure they load
from deepforest import tfrecords
import tensorflow as tf
import glob
import os
    
def check_shape():
    list_of_tfrecords = glob.glob("/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/tfrecords/*.tfrecord")
    inputs, targets = tfrecords.create_tensors(list_of_tfrecords,shuffle=False, repeat=False)
    
    sess = tf.Session()
    
    counter = 0
    while True:
        try:
            tf_inputs, tf_targets = sess.run([inputs,targets])
            assert tf_inputs.shape == (1,800,800,3)
            assert tf_targets[0].shape[2] == 5
            counter+=1
        except tf.errors.OutOfRangeError as e:
            print("Tensor completed")
            break
        except tf.errors.InvalidArgumentError as e:
            print("{} index has shape error".format(counter))
    
    print(counter)

def _cast_fn(example):
    # Define features
    features = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        "image/object/regression_target": tf.VarLenFeature(tf.float32),
        "image/object/class_target": tf.VarLenFeature(tf.int64),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/target_height": tf.FixedLenFeature([], tf.int64),
        "image/target_width": tf.FixedLenFeature([], tf.int64),
        "image/object/n_anchors": tf.FixedLenFeature([], tf.int64)
    }
    # Load one example and parse
    example = tf.io.parse_single_example(example, features)
    # Load image from file
    filename = tf.cast(example["image/filename"], tf.string)
    loaded_image = tf.read_file(filename)
    loaded_image = tf.image.decode_image(loaded_image, 3)
    # Reshape to known shape
    image_rows = tf.cast(example['image/height'], tf.int32)
    image_cols = tf.cast(example['image/width'], tf.int32)
    #Wrap in a try catch and report file failure, this can be not graceful exiting
    loaded_image = tf.reshape(loaded_image,
                          tf.stack([image_rows, image_cols, 3]),
                          name="cast_loaded_image")
    # needs to be float to subtract weights below
    loaded_image = tf.cast(loaded_image, tf.float32)
    # Turn loaded image from rgb into bgr and subtract imagenet means, see keras_retinanet.utils.image.preprocess_image
    red, green, blue = tf.unstack(loaded_image, axis=-1)
    # Subtract imagenet means
    blue = tf.subtract(blue, 103.939)
    green = tf.subtract(green, 116.779)
    red = tf.subtract(red, 123.68)
    # Recombine as BGR image
    loaded_image = tf.stack([blue, green, red], axis=-1)
    # Resize loaded image to desired target shape
    target_height = tf.cast(example['image/target_height'], tf.int32)
    target_width = tf.cast(example['image/target_width'], tf.int32)
    loaded_image = tf.image.resize(loaded_image, (target_height, target_width), align_corners=True)
    # Generated anchor data
    regression_target = tf.sparse_tensor_to_dense(example['image/object/regression_target'])
    class_target = tf.sparse_tensor_to_dense(example['image/object/class_target'])
    target_n_anchors = tf.cast(example['image/object/n_anchors'], tf.int32)
    regression_target = tf.cast(regression_target, tf.float32)
    class_target = tf.cast(class_target, tf.float32)
    regression_target = tf.reshape(regression_target, [target_n_anchors, 5], name="cast_regression")
    class_target = tf.reshape(class_target, [target_n_anchors, 2], name="cast_class_label")
    return filename

#Find corrupt files
def tfrecord_filenames(path="/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/tfrecords/*.tfrecord"):
    list_of_tfrecords = glob.glob(path)    
    dataset = tf.data.TFRecordDataset(list_of_tfrecords)
    #Create dataset and filter out errors
    dataset=dataset.map(_cast_fn)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())    
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    good_filenames = []
    counter = 0
    while True:
        try:
            f = sess.run(next_element)
            good_filenames.append(f.decode("utf-8") )
            counter +=1
        except tf.errors.OutOfRangeError as e:
            print("Tensor completed")
            break
        except tf.errors.InvalidArgumentError as e:
            print("{} index has shape error".format(counter))
    #return as just basenames
    return good_filenames

def original_filenames(path="/orange/ewhite/b.weinstein/NeonTreeEvaluation/hand_annotations/crops/"):
    fils = glob.glob(path+"*.png")
    return([os.path.basename(x) for x in fils])


if __name__ == "__main__":
    check_shape()
    print("running missing file check")
    tfnames  = tfrecord_filenames()
    tfbasename = [os.path.basename(x) for x in tfnames]
    print("There are {} valid values in the tfrecords".format(len(tfbasename)))
    original = original_filenames()
    print("There are {} cropped images".format(len(original)))
    missing_files = [x for x in original if x not in tfbasename]
    print("There are {} missing files".format(len(missing_files)))
    missing_files