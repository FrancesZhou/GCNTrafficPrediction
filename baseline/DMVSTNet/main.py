import numpy as np
import sys
import os
import argparse
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import EarlyStopping, ModelCheckpoint
currentPath = os.path.abspath(os.path.curdir)
sys.path.append(currentPath)
from .model import build_model
from .utils import *
from .minMax import *


def load_embedding(filename):
    with open(filename, 'r') as f:
        num, length = f.readline().strip().split(" ")
        embedding = {}
        for i in range(int(num)):
            line = f.readline()
            vector = line.strip().split(" ")
            id = int(vector[0])
            vector = [float(item) for item in vector[1:]]
            embedding[id] = np.array(vector, dtype=np.float)
    return embedding


def prepare_data(input_length, map_data, embedding_data, splits, image_size=9):
    height = map_data.shape[1]
    width = map_data.shape[2]
    #T = map_data.shape[0]
    train_image_x = []
    train_y = []
    train_embedding = []
    test_image_x = []
    test_y = []
    test_embedding_x = []
    padding = int(image_size / 2)
    #for t in range(input_length, splits[0] - input_length):
    for t in range(input_length, splits[0]):
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                if not embedding_data.__contains__(i * width + j):
                    continue
                train_image_x.append(map_data[t-input_length:t, i - padding:i + padding+1, j - padding:j + padding+1, :])
                train_y.append(map_data[t, i, j, :])
                train_embedding.append(embedding_data[i * width + j])
    #for t in range(splits[0] - input_length, splits[0] + splits[1] - input_length):
    for t in range(np.sum(split[:-1]) + input_length, np.sum(split)):
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                if not embedding_data.__contains__(i * width + j):
                    continue
                test_image_x.append(map_data[t-input_length:t, i - padding:i + padding+1, j - padding:j + padding+1, :])
                test_y.append(map_data[t, i, j, :])
                test_embedding_x.append(embedding_data[i * width + j])
    train_image_x = np.asarray(train_image_x)
    train_embedding = np.asarray(train_embedding)
    train_y = np.asarray(train_y)
    test_embedding_x = np.asarray(test_embedding_x)
    test_image_x = np.asarray(test_image_x)
    test_y = np.asarray(test_y)
    print(train_image_x.shape)
    print(train_embedding.shape)
    print(train_y.shape)
    print(test_image_x.shape)
    print(test_embedding_x.shape)
    print(test_y.shape)
    return train_image_x, train_embedding, train_y, test_image_x, test_embedding_x, test_y

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    parse.add_argument('-dataset', '--dataset', type=str, default='didi', help='datasets: didi, taxi')
    # parse.add_argument('-dataset', '--dataset', type=str, default='taxi')
    #
    parse.add_argument('-predict_steps', '--predict_steps', type=int, default=1, help='prediction steps')
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6, help='number of input steps')
    parse.add_argument('-dim', '--dim', type=int, default=0, help='dim of data to be processed')
    #
    args = parse.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #
    data_folder = '../../datasets/' + args.dataset + '-data/data/'
    if args.dim > 0:
        embedding_file = os.path.join('./data/' + args.dataset, 'dim1', 'embedding.txt')
    else:
        embedding_file = os.path.join('./data/'+args.dataset, 'embedding.txt')
    print('load train, test data...')
    if 'citibike' in args.dataset:
        split = [3672, 240, 480]
        data, train_data, _, test_data = load_npy_data(
            filename=[data_folder + 'd_station.npy', data_folder + 'p_station.npy'], split=split)
    elif 'taxi' in args.dataset:
        split = [11640, 744, 720]
        data_folder = '../../datasets/' + args.dataset + '-data/graph-data/'
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'nyc_taxi_data.npy'], split=split)
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 10, -1])
    elif 'didi' in args.dataset:
        split = [2400, 192, 288]
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'cd_didi_data.npy'], split=split)
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 20, -1])

    #
    data = data[..., args.dim]
    minMax = MinMax(data)
    data = minMax.transform()
    embedding = load_embedding(embedding_file)
    # nyb_splits = [data.shape[0]-240, 240]
    # nyt_splits = [data.shape[0] - 720, 720]
    # didi_splits = [data.shape[0] - 288, 288]
    # prepare data
    train_image, train_embedding, train_y, test_image, test_embedding, test_y = prepare_data(args.input_steps,
                                                                                             data, embedding,
                                                                                             split, 9)
    # set gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ktf.set_session(tf.Session(config=config))
    # train model
    model = build_model(train_y, test_y, train_image, test_image, train_embedding, test_embedding, 64, minMax,
                        seq_len=args.input_steps, trainable=True)
