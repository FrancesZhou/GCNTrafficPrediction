import numpy as np
import sys
import os
import argparse
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.callbacks import EarlyStopping, ModelCheckpoint
currentPath = os.path.abspath(os.path.curdir)
sys.path.append(currentPath)
from model import build_model
from utils import *
from minMax import *


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


def prepare_data_padding(input_length, map_data, embedding_data, split, image_size=9, if_padding=False):
    padding = int(image_size / 2)
    #
    if if_padding:
        raw_map_shape = map_data.shape
        padding_map_data = np.zeros((raw_map_shape[0], raw_map_shape[1]+2*padding, raw_map_shape[2]+2*padding, raw_map_shape[3]), dtype=np.float32)
        padding_map_data[:, padding:-padding, padding:-padding, :] = map_data
        map_data = padding_map_data
    #
    height = map_data.shape[1]
    width = map_data.shape[2]
    #T = map_data.shape[0]
    train_image_x = []
    train_y = []
    train_embedding = []
    test_image_x = []
    test_y = []
    test_embedding_x = []
    #for t in range(input_length, splits[0] - input_length):
    for t in range(input_length, split[0]):
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
    test_y_num = [np.sum(split) - np.sum(split[:-1]) - input_length]
    return train_image_x, train_embedding, train_y, test_image_x, test_embedding_x, test_y, test_y_num


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    parse.add_argument('-dataset', '--dataset', type=str, default='didi', help='datasets: didi, taxi')
    # parse.add_argument('-dataset', '--dataset', type=str, default='taxi')
    #
    parse.add_argument('-model_save', '--model_save', type=str, default='', help='path to save model')
    parse.add_argument('-predict_steps', '--predict_steps', type=int, default=1, help='prediction steps')
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6, help='number of input steps')
    parse.add_argument('-dim', '--dim', type=int, default=0, help='dim of data to be processed')
    parse.add_argument('-trainable', '--trainable', type=int, default=1, help='if to train (1) or to test (0)')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=64)
    #
    args = parse.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #
    data_folder = '../../datasets/' + args.dataset + '-data/data/'
    if args.dim > 0:
        embedding_file = os.path.join('./data', args.dataset, 'dim1', 'embedding.txt')
        output_folder = os.path.join('./data', args.dataset, 'dim1', 'model_save', args.model_save)
    else:
        embedding_file = os.path.join('./data', args.dataset, 'embedding.txt')
        output_folder = os.path.join('./data', args.dataset, 'model_save', args.model_save)
    #
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #
    print('load train, test data...')
    if 'taxi' in args.dataset:
        #split = [11640, 744, 720]
        split = [11640+744, 720]
        data_folder = '../../datasets/' + args.dataset + '-data/graph-data/'
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'nyc_taxi_data.npy'], split=split)
        data = np.reshape(data, [data.shape[0], 20, 10, -1])
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 10, -1])
    elif 'didi' in args.dataset:
        #split = [2400, 192, 288]
        split = [2400+192, 288]
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'cd_didi_data.npy'], split=split)
        data = np.reshape(data, [data.shape[0], 20, 20, -1])
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 20, -1])

    #
    print(data.shape)
    data = data[..., args.dim]
    data = np.expand_dims(data, axis=-1)
    print(data.shape)
    minMax = MinMaxNormalization01()
    minMax.fit(data[:split[0]])
    data = minMax.transform(data)
    #minMax = MinMax(data)
    #data = minMax.transform()
    embedding = load_embedding(embedding_file)
    # prepare data
    train_image, train_embedding, train_y, test_image, test_embedding, test_y, test_num = prepare_data_padding(args.input_steps,
                                                                                             data, embedding,
                                                                                             split, 9, if_padding=True)
    print(test_num)
    # set gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ktf.set_session(tf.Session(config=config))
    # train model
    prediction = build_model(train_y, test_y, train_image, test_image, train_embedding, test_embedding, 64, minMax,
                             seq_len=args.input_steps,
                             batch_size=args.batch_size,
                             trainable=args.trainable,
                             model_path=output_folder)
    test_target = np.reshape(test_y, test_num+[-1])
    test_prediction = np.reshape(prediction, test_num+[-1])
    np.save(os.path.join(output_folder, 'test_target.npy'), test_target)
    np.save(os.path.join(output_folder, 'test_prediction.npy'), test_prediction)
    
    
if __name__ == '__main__':
    main()
