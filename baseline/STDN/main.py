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



def prepare_data_padding(input_length, map_data, flow_data, split, nb_size=7):
    padding = int(nb_size / 2)
    #
    raw_map_shape = map_data.shape
    padding_map_data = np.zeros((raw_map_shape[0], raw_map_shape[1]+2*padding, raw_map_shape[2]+2*padding, raw_map_shape[3]), dtype=np.float32)
    padding_map_data[:, padding:-padding, padding:-padding, :] = map_data
    #
    height = padding_map_data.shape[1]
    width = padding_map_data.shape[2]
    index = np.ones((height, width), dtype=np.int32) * (-1)
    index[padding:-padding, padding:-padding] = np.reshape(np.arange(raw_map_shape[1]*raw_map_shape[2]), (raw_map_shape[1], raw_map_shape[2]))
    #
    train_image_x = []
    train_y = []
    train_flow = []
    test_image_x = []
    test_y = []
    test_flow = []
    for t in range(input_length, split[0]):
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                train_image_x.append(map_data[t-input_length:t, i - padding:i + padding+1, j - padding:j + padding+1, :])
                train_y.append(map_data[t, i, j, :])
                #
                ij_index = i * raw_map_shape[2] + j
                nb_index = index[i-padding: i+padding+1, j-padding:j+padding+1].flatten()
                ij_flow = flow_data[t, ij_index, nb_index]
                ij_flow[nb_index<0] = 0
                ij_flow = np.reshape(ij_flow, (nb_size, nb_size))
                train_flow.append(ij_flow)
    for t in range(np.sum(split[:-1]) + input_length, np.sum(split)):
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                test_image_x.append(map_data[t-input_length:t, i - padding:i + padding+1, j - padding:j + padding+1, :])
                test_y.append(map_data[t, i, j, :])
                ij_index = i * raw_map_shape[2] + j
                nb_index = index[i - padding: i + padding + 1, j - padding:j + padding + 1].flatten()
                ij_flow = flow_data[t, ij_index, nb_index]
                ij_flow[nb_index < 0] = 0
                ij_flow = np.reshape(ij_flow, (nb_size, nb_size))
                test_flow.append(ij_flow)
    train_image_x = np.asarray(train_image_x)
    train_flow = np.asarray(train_flow)
    train_y = np.asarray(train_y)
    test_flow = np.asarray(test_flow)
    test_image_x = np.asarray(test_image_x)
    test_y = np.asarray(test_y)
    print(train_image_x.shape)
    print(train_flow.shape)
    print(train_y.shape)
    print(test_image_x.shape)
    print(test_flow.shape)
    print(test_y.shape)
    test_y_num = [np.sum(split) - np.sum(split[:-1]) - input_length]
    return train_image_x, train_flow, train_y, test_image_x, test_flow, test_y, test_y_num


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
        #
        f_data, train_f_data, val_f_data, test_f_data = load_npy_data([data_folder + 'nyc_taxi_flow_in.npy'],
                                                                      split=split)
    elif 'didi' in args.dataset:
        #split = [2400, 192, 288]
        split = [2400+192, 288]
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'cd_didi_data.npy'], split=split)
        data = np.reshape(data, [data.shape[0], 20, 20, -1])
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 20, -1])
        #
        f_data, train_f_data, val_f_data, test_f_data = load_npy_data([data_folder + 'cd_didi_flow_in.npy'],
                                                                      split=split)

    #
    print('preprocess data...')
    print('data shape: {}'.format(data.shape))
    minMax = MinMaxNormalization01()
    minMax.fit(train_data)
    data = minMax.transform(data)
    #
    print('preprocess f_data...')
    print('f_data shape: {}'.format(f_data.shape))
    f_minMax = MinMaxNormalization01()
    f_minMax.fit(train_f_data)
    f_data = f_minMax.transform(f_data)
    # prepare data
    train_x, train_flow, train_y, test_x, test_flow, test_y, test_num = prepare_data_padding(args.input_steps,
                                                                                             data, f_data,
                                                                                             split, 7, if_padding=True)
    print(test_num)
    # set gpu config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ktf.set_session(tf.Session(config=config))
    # train model
    prediction = build_model(train_y, test_y, train_x, test_x, train_flow, test_flow, minMax,
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
