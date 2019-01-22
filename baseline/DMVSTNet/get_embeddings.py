import os
import numpy as np
import argparse
from .utils import *
from dtw import dtw
#from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
import pandas as pd


def getGraphEmbedding(data, week_length=24 * 7, output_folder='./'):
    """
    :param data: t x w x h x c
    :return:
    """
    #embedding = None
    graph = np.zeros([data.shape[1] * data.shape[2], data.shape[1] * data.shape[2]], dtype=np.float32)
    s = np.zeros([week_length, data.shape[1], data.shape[2]])
    count = 0
    for i in range(0, data.shape[0], week_length):
        if i + week_length > data.shape[0]:
            break
        s += data[i:i + week_length, :, :, 0]
        count += 1
    s = s / count
    s = np.reshape(s, [s.shape[0], -1])
    for i in range(s.shape[1]):
        for j in range(i, s.shape[1]):
            # if np.sum(s[:, i]) > 1440 and np.sum(s[:, j]) > 1440:
            #     graph[i, j], cost, acc, path = dtw(s[:, i], s[:, j], dist=lambda x, y: abs(x - y))
            graph[i, j], cost, acc, path = dtw(s[:, i], s[:, j], dist=lambda x, y: abs(x - y))
            print("at %d %d " % (i, j))
    np.save(os.path.join(output_folder, 'graph.npy'), graph)
    # graph = np.load("./graph.npy")
    with open(os.path.join(output_folder, 'graph_embedding_input.txt'), 'w') as f:
        for i in range(graph.shape[0]):
            for j in range(i, graph.shape[1]):
                f.write(str(i) + " " + str(j) + " " + str(graph[i, j]) + "\n")
                f.write(str(j) + " " + str(i) + " " + str(graph[i, j]) + "\n")


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-dataset', '--dataset', type=str, default='didi')
    # parse.add_argument('-dataset', '--dataset', type=str, default='citibike')
    # parse.add_argument('-dataset', '--dataset', type=str, default='taxi')
    parse.add_argument('-predict_steps', '--predict_steps', type=int, default=1, help='prediction steps')
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6, help='number of input steps')
    parse.add_argument('-dim', '--dim', type=int, default=0, help='dim of data to be processed')
    #
    args = parse.parse_args()
    #
    data_folder = '../../datasets/' + args.dataset + '-data/data/'
    output_folder = './data/' + args.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print('load train, test data...')
    if 'citibike' in args.dataset:
        split = [3672, 240, 480]
        data, train_data, _, test_data = load_npy_data(
            filename=[data_folder + 'd_station.npy', data_folder + 'p_station.npy'], split=split)
        p = 24 * 7
    elif 'taxi' in args.dataset:
        split = [11640, 744, 720]
        data_folder = '../datasets/' + args.dataset + '-data/graph-data/'
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'nyc_taxi_data.npy'], split=split)
        p = 24 * 7
    elif 'didi' in args.dataset:
        split = [2400, 192, 288]
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'cd_didi_data.npy'], split=split)
        p = 24 * 7 * 4
    print(data.shape)
    train_emb_data = train_data[..., args.dim]
    getGraphEmbedding(train_emb_data, week_length=p, output_folder=output_folder)


if __name__ == '__main__':
    main()