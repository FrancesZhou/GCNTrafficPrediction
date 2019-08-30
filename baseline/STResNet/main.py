import numpy as np
import sys
import os
import argparse
#import tensorflow as tf
#from ResNet import ResNet
from preprocessing import *
from utils import *



def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    parse.add_argument('-dataset', '--dataset', type=str, default='didi', help='datasets: didi, taxi')
    # parse.add_argument('-dataset', '--dataset', type=str, default='taxi')
    #
    parse.add_argument('-closeness', '--closeness', type=int, default=3, help='num of closeness')
    parse.add_argument('-period', '--period', type=int, default=4, help='num of period')
    parse.add_argument('-trend', '--trend', type=int, default=4, help='num of trend')
    #
    parse.add_argument('-model_save', '--model_save', type=str, default='', help='path to save model')
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6, help='number of input steps')
    parse.add_argument('-dim', '--dim', type=int, default=0, help='dim of data to be processed')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=8)
    #
    parse.add_argument('-train', '--train', type=int, default=0, help='whether to train')
    parse.add_argument('-test', '--test', type=int, default=0, help='whether to train')
    parse.add_argument('-save_results', '--save_results', type=int, default=1, help='whether to train')
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
        split = [11640, 744, 720]
        #split = [11640+744, 720]
        data_folder = '../../datasets/' + args.dataset + '-data/graph-data/'
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'nyc_taxi_data.npy'], split=split)
        data = np.reshape(data, [data.shape[0], 20, 10, -1])
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 10, -1])
        all_timestamps = gen_timestamps(['2014', '2015'])
        all_timestamps = all_timestamps[:-4416]
        #
    elif 'didi' in args.dataset:
        split = [2400, 192, 288]
        #split = [2400+192, 288]
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'cd_didi_data.npy'], split=split)
        data = np.reshape(data, [data.shape[0], 20, 20, -1])
        train_data = np.reshape(train_data, [train_data.shape[0], 20, 20, -1])
        all_timestamps = gen_timestamps_intervals('2016', 11, 15)
        #

    #
    print('preprocess data...')
    print('data shape: {}'.format(data.shape))
    minMax = MinMaxNormalization01()
    minMax.fit(train_data)
    data = minMax.transform(data)
    #
    pre_index = max(args.closeness * 1, args.period * 7, args.trend * 7 * 24)
    # train_data = train_data
    train_data = data[:split[0]]
    val_data = data[split[0] - pre_index:split[0] + split[1]]
    test_data = data[split[0] + split[1] - pre_index:split[0] + split[1] + split[2]]
    # get train, validate, test timestamps
    train_timestamps = all_timestamps[:split[0]]
    val_timestamps = all_timestamps[split[0] - pre_index:split[0] + split[1]]
    test_timestamps = all_timestamps[split[0] + split[1] - pre_index:split[0] + split[1] + split[2]]
    # get x, y
    train_x, train_y = batch_data_cpt_ext(train_data, train_timestamps,
                                          batch_size=args.batch_size, close=args.closeness, period=args.period,
                                          trend=args.trend)
    val_x, val_y = batch_data_cpt_ext(val_data, val_timestamps,
                                      batch_size=args.batch_size, close=args.closeness, period=args.period,
                                      trend=args.trend)
    test_x, test_y = batch_data_cpt_ext(test_data, test_timestamps,
                                        batch_size=args.batch_size, close=args.closeness, period=args.period,
                                        trend=args.trend)
    train = {'x': train_x, 'y': train_y}
    val = {'x': val_x, 'y': val_y}
    test = {'x': test_x, 'y': test_y}
    nb_flow = train_data.shape[-1]
    row = train_data.shape[1]
    col = train_data.shape[2]

    print('build ResNet model...')
    model_path = output_folder
    log_path = os.path.join(output_folder, '/log')
    model = ResNet(input_conf=[[args.closeness, nb_flow, row, col], [args.period, nb_flow, row, col],
                               [args.trend, nb_flow, row, col], [8]], batch_size=args.batch_size,
                   layer=['conv', 'res_net', 'conv'],
                   layer_param=[[[3, 3], [1, 1, 1, 1], 64],
                                [3, [[[3, 3], [1, 1, 1, 1], 64], [[3, 3], [1, 1, 1, 1], 64]]],
                                [[3, 3], [1, 1, 1, 1], 2]])
    print('model solver...')
    solver = ModelSolver(model, train, val, preprocessing=minMax,
                         n_epochs=args.n_epochs,
                         batch_size=args.batch_size,
                         update_rule=args.update_rule,
                         learning_rate=args.lr, save_every=args.save_every,
                         pretrained_model=args.pretrained_model, model_path=model_path,
                         test_model='citibike-results/model_save/ResNet/model-' + str(args.n_epochs),
                         log_path=log_path,
                         cross_val=False, cpt_ext=True)
    if args.train:
        print('begin training...')
        test_prediction = solver.train(test)
        test_target = test['y']
        if args.save_results:
            save_path = os.path.join(output_folder, '/results/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + 'test_target.npy', test_target)
            np.save(save_path + 'test_prediction.npy', test_prediction)
    if args.test:
        print('begin testing...')
        solver.test_model = solver.model_path + args.pretrained_model
        solver.test(test)
        test_target = test['y']
        if args.save_results:
            save_path = os.path.join(output_folder, '/results/')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.save(save_path + 'test_target.npy', test_target)
            np.save(save_path + 'test_prediction.npy', test_prediction)


if __name__ == '__main__':
    main()
