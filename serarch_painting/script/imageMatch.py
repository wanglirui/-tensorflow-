#
# Author: Philipp Jaehrling
# Influenced by:
# - https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html
# - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
#

import os
import fnmatch
import argparse
import numpy as np
import tensorflow as tf

from models.alexnet import AlexNet
from models.vgg_data import VGG
from models.vgg_slim import VGGslim
from models.inception_v3 import InceptionV3
from models.resnet_v2 import ResNetV2
from helper.imagenet_classes import class_names
import math
from pymongo import MongoClient
import requests as req
from io import BytesIO
from PIL import Image
import time
from elasticsearch import Elasticsearch

def countdistance(list1, list2):
    a = list1 * list2
    b = list1 * list1
    c = list2 * list2
    return a.sum() / (math.sqrt(b.sum()) * math.sqrt(c.sum()))

def getMap(matrix,len=256,k=16):
    vec = []
    temp = []
    for i in range(k):
        vec.append(math.pow(3, i))
    vec = np.array(vec)
    for i in range(matrix.shape[0]):
        if matrix[i] > 0:
            temp.append(1)
        else:
            temp.append(0)
    temp = np.array(temp)

    temp = temp.reshape(len,k)
    result = np.dot(temp, vec.T).tolist()
    return result




#from tensorflow.contrib.data import Dataset, Iterator

def wrateToDb(dict):
    client = MongoClient('115.231.226.46', 27017)
    db_auth = client.dlsdata_v2
    db_auth.authenticate("pgc", "GPEGssLpHP04")
    db = client.dlsdata_v2
    collection = db.painting_cnn_features
    collection.insert(dict)

def readTxt(filename,model="r"):
    fr = open(filename,model)
    data = []
    for line in fr.readlines():
        line = line.strip(" ").split("\t")
        data.append(line)
    return data

def prep_resnet_results(probs):
    """
    For ResNet the result is a rank-4 tensor of size [images, 1, 1, num_classes].
    """
    return [prob[0][0] for prob in probs]


def validate(model_def,begin,end,floor):
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
                print("load %d images",ind)
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
    next_element = iterator.get_next()
    iterator_init_op = iterator.make_initializer(images)

    # create the model and get scores (pipe to softmax)
    model = model_def(next_element)
    endpoints = model.get_endpoints()



    print('start validation ...')


    with tf.Session() as sess:

        # Initialize all variables and the iterator
        sess.run(tf.global_variables_initializer())
        sess.run(iterator_init_op)

        # Load the pretrained weights into the model
        model.load_initial_weights(sess)

        # run the graph
        endpoints = sess.run(endpoints)
        scores = endpoints[floor]
        return scores


def main():
    #es = Elasticsearch(['115.231.226.151:10200'])
    es = Elasticsearch(['10.4.40.181:9200'])

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
        default=100000,
        help='Model to be validated'
    )
    parser.add_argument(
        '-index',
        type=str,
        default='painting_search_cnn_new',
        help='Model to be validated'
    )
    parser.add_argument(
        '-floor',
        type=str,
        default="fc7",
        help='Model to be validated'
    )
    # 默认使用painting_search_cnn_new，fc7也可以改成painting_search_cnn和fc8
    args = parser.parse_args()
    model_str = args.model
    begin = args.begin
    end = args.end
    index = args.index
    floor = args.floor

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

    scores = validate(model_def,begin,end,floor)

    score = scores[3]


    #使用poolfc7产生的索引，参数index改成painting_search_cnn_v2，把下面的注释去掉

    '''
    temp = []
    for i in range(int(len(score) / 4)):
        max = 0.0
        for j in range(i * 4, i * 4 + 4):
            if score[j] > max:
                max = score[j]
        temp.append(max)
    pool = np.array(temp)

    score = pool

    mapScore = getMap(score, 256, 4)
    '''

    mapScore = getMap(score)
    for i in range(len(mapScore)):
        print(mapScore[i])

    data = {}
    should = []
    for i in range(len(mapScore)):
        match = {}
        simple = {}
        strname = "simple_word_" + str(i)
        simple[strname] = int(mapScore[i])
        match["match"] = simple
        should.append(match)
    bools = {}
    bools["should"] = should
    query = {}
    query["bool"] = bools
    data["query"] = query
    print(data)
    # s = time.time()
    res = es.search(index=index, doc_type=index, body=data, size=100)
    # print("select time: %s" % (time.time() - s))

    hits = res["hits"]["hits"]

    sortres = {}

    for i in range(100):
        id = hits[i]["_source"]["url"]
        #id = hits[i]["_source"]["id"]
        if id not in sortres.keys():
            sortres[id] = countdistance(score, np.array(hits[i]["_source"]["signature"]))
        else:
            if countdistance(score, np.array(hits[i]["_source"]["signature"])) > sortres[id]:
                sortres[id] = countdistance(score, np.array(hits[i]["_source"]["signature"]))

    for i in range(10):
        max = 0.0
        for key in sortres.keys():
            if sortres[key] > max:
                max = sortres[key]
                maxkey = key
        print(maxkey)
        print(sortres[maxkey])
        #response = req.get(maxkey)
        #im = Image.open(BytesIO(response.content))
        #resize_path = "./selectimage/" + maxkey.split('/')[-1]
        #im.save(resize_path)
        del sortres[maxkey]

if __name__ == '__main__':
    main()