#
# Author: Philipp Jaehrling
# Influenced by:
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import os
import argparse
import numpy as np
import tensorflow as tf
from my_models.alexnet import AlexNet
from my_models.vgg_data import VGG
from my_models.vgg_slim import VGGslim
from my_models.inception_v3 import InceptionV3
from my_models.resnet_v2 import ResNetV2
from pymongo import MongoClient




#from tensorflow.contrib.data import Dataset, Iterator

def wrateToDb(dict):
    #client = MongoClient('115.231.226.46', 27017)
    #db_auth = client.dlsdata_v2
    #db_auth.authenticate("pgc", "GPEGssLpHP04")
    #db = client.dlsdata_v2
    #collection = db.painting_cnn_new_features

    client = MongoClient('10.4.40.129', 27017)
    db = client.dlsdata_v2
    collection = db.painting_vgg16_features512_672
    collection.insert(dict)

def readTxt(filename,model="r"):
    fr = open(filename,model)
    data = []
    for line in fr.readlines():
        line = line.strip(" ").strip("\n").split("\t")
        data.append(line)
    return data

def prep_resnet_results(probs):
    """
    For ResNet the result is a rank-4 tensor of size [images, 1, 1, num_classes].
    """
    return [prob[0][0] for prob in probs]


def validate(model_def,begin,end):
    """
    Validate my alexnet implementation

    Args:
        model_def: the model class/definition
    """

    img_dir = os.path.join('..', 'images')
    images = []

    print("loading images ...")
    #files = os.listdir(img_dir)
    files = readTxt("../images/data.txt")
    ind = -1
    for f in files:
        ind += 1
        if begin <= ind <= end:
            f = f[0]
            #print("> " + f)
            #print(os.path.join(img_dir, f))\
            if ind % 100 == 0:
                print("load %d images" % ind)
            img_file      = tf.read_file(os.path.join(img_dir, f))
            img_decoded   = tf.image.decode_jpeg(img_file, channels=3)
            img_processed = model_def.image_prep.preprocess_image(
                image=img_decoded,
                output_height=model_def.image_size,
                output_width=model_def.image_size,
                is_training=False
            )
            images.append(img_processed)

    # create TensorFlow Iterator object

    images = tf.data.Dataset.from_tensors(images)
    iterator = tf.data.Iterator.from_structure(images.output_types, images.output_shapes)
    print(images.output_shapes)
    print(images.output_shapes)
    next_element = iterator.get_next()
    iterator_init_op = iterator.make_initializer(images)

    # create the model and get scores (pipe to softmax)
    model = model_def(next_element)
    endpoints = model.get_endpoints()



    print('start validation ...')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 占用GPU90%的显存

    with tf.Session(config=config) as sess:

        # Initialize all variables and the iterator
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_init_op)

        # Load the pretrained weights into the model
        model.load_initial_weights(sess)
        # saver = tf.train.Saver()
        # saver.save(sess, "../models/weights/my_model.ckpt")
        # run the graph
        endpoints = sess.run(endpoints)
        scores = endpoints['pool5']
        scores=np.mean(scores,1)
        scores=np.mean(scores,1)
        # print(type(scores))
        # print(np.shape(scores))
        # print(np.size(scores, 1))
        # print(scores)
        # print(scores[1])
        for i in range(len(scores)):
            data = {}
            data['imageId'] = files[i+begin][1]
            data['imageName'] = files[i+begin][0]
            data['features'] = scores[i].tolist()
            wrateToDb(data)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-model',
        type=str,
        default='vgg',
        help='Model to be validated'
    )
    parser.add_argument(
        '-begin',
        type=int,
        default=0,
        help='Model to be validated'
    )
    parser.add_argument(
        '-end',
        type=int,
        default=500000,
        help='Model to be validated'
    )
    args = parser.parse_args()
    model_str = args.model
    begin = args.begin
    end = args.end

    if model_str == 'vgg':
        model_def = VGG
    elif model_str == 'vgg_slim':
        model_def = VGGslim
    elif model_str == 'inc_v3':
        model_def = InceptionV3
    elif model_str == 'res_v2':
        model_def = ResNetV2
    else: # default
        model_def = AlexNet

    validate(model_def,begin,end)

if __name__ == '__main__':
    main()