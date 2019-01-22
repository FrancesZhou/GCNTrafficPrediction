from __future__ import print_function
import numpy as np
import warnings
import os
import argparse
import sys
import pickle
#from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import ARIMA
sys.path.append('../')
from utils import *


def predict_by_samples(data, test_data, train_length, num_sample, output_steps, lag_order=1, if_sample=True):
    warnings.filterwarnings("ignore")
    test_num, data_dim = test_data.shape
    if not if_sample:
        num_sample = data_dim
    #widgets = ['Train: ', Percentage(), ' ', Bar('-'), ' ', ETA()]
    #pbar = ProgressBar(widgets=widgets, maxval=test_data.shape[0]-output_steps).start()
    error_all = []
    index_all = np.zeros([test_num - output_steps, num_sample], dtype=np.int32)
    valid_num = np.zeros(test_num - output_steps, dtype=np.int32)
    real = np.zeros([test_num - output_steps, num_sample])
    predict = np.zeros([test_num - output_steps, num_sample])
    #
    samples = np.arange(data_dim)
    #
    for t in range(test_num - output_steps):
        #pbar.update(t)
        if t%100 == 0:
            print(t)
        error_index = []
        if if_sample:
            samples = np.random.randint(data_dim, size=num_sample)
        for r in range(num_sample):
            # t: which time slot
            # i: which station
            i = samples[r]
            train_df = data[t:train_length+t, i]
            try:
                results = ARIMA(train_df, order=(lag_order, 0, 1)).fit(trend='nc', disp=0)
            except:
                error_index.append(r)
                continue
            #pre, _, _ = results.forecast(output_steps)
            #pre = results.predict(train_length, train_length+output_steps)
            pre = results.predict(train_length, train_length)
            test_real = test_data[i][t:t + output_steps]
            real[t, r] = test_real
            #print(pre)
            predict[t, r] = pre
        index_all[t] = samples
        error_all.append(error_index)
        valid_num[t] = num_sample - len(error_index)
    #pbar.finish()
    return real, predict, index_all, error_all, valid_num


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument('-dataset', '--dataset', type=str, default='didi')
    # parse.add_argument('-dataset', '--dataset', type=str, default='citibike')
    parse.add_argument('-dataset', '--dataset', type=str, default='taxi')
    parse.add_argument('-predict_steps', '--predict_steps', type=int, default=1, help='prediction steps')
    parse.add_argument('-lag_order', '--lag_order', type=int, default=1, help='lag order in VAR and ARIMA models')
    parse.add_argument('-num_samples', '--num_samples', type=int, default=5, help='number of samples for ARIMA model')
    #
    args = parse.parse_args()
    #
    data_folder = '../datasets/' + args.dataset + '-data/data/'
    output_folder = 'results/ARIMA/' + args.dataset
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    prediction_file_name = output_folder + 'ARIMA_prediction.npy'
    target_file_name = output_folder + 'ARIMA_target.npy'
    print('load train, test data...')
    if 'citibike' in args.dataset:
        split = [3672, 240, 480]
        data, train_data, _, test_data = load_npy_data(filename=[data_folder+'d_station.npy', data_folder+'p_station.npy'], split=split)
    elif 'taxi' in args.dataset:
        split = [11640, 744, 720]
        data_folder = '../datasets/' + args.dataset + '-data/graph-data/'
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'nyc_taxi_data.npy'], split=split)
    elif 'didi' in args.dataset:
        split = [2400, 192, 288]
        data, train_data, val_data, test_data = load_npy_data(filename=[data_folder + 'cd_didi_data.npy'], split=split)
    s = data.shape
    print(s)
    if 0:
    #if os.path.exists(prediction_file_name):
        test_predict = np.load(prediction_file_name)
        test_real = np.load(target_file_name)
        error_all = load_pickle(output_folder+'arima_error.pkl')
        valid_num = np.array([args.num_samples - len(e) for e in error_all])
    else:
        data = np.reshape(data, (data.shape[0], -1))
        test_data = np.reshape(test_data, (test_data.shape[0], -1))
        #
        print('train ARIMA model...')
        #test_data_preindex = np.vstack((train_data[-args.lag_order:], test_data))
        test_real, test_predict, index_all, error_all, valid_num = predict_by_samples(data, test_data, split[0]+split[1], args.num_samples, args.predict_steps, lag_order=args.lag_order)
        test_predict = np.clip(test_predict, 0, None)
        test_real = np.squeeze(test_real)
        test_predict = np.squeeze(test_predict)
        #
        test_predict = np.array(test_predict, dtype=np.float32)
        test_real = np.array(test_real, dtype=np.float32)
        np.save(prediction_file_name, test_predict)
        np.save(target_file_name, test_real)
        np.save(output_folder + 'index_all.npy', index_all)
        dump_pickle(error_all, output_folder + 'arima_error.pkl')
    #
    #rmse_test = np.sqrt(np.sum(np.square(test_real-test_predict))/np.prod(test_predict.shape))
    rmse_test = np.sqrt(np.sum(np.square(test_real - test_predict)) / np.sum(valid_num))
    print('test in/out rmse is %.4f' % rmse_test)
    #
    # if not os.path.exists(prediction_file_name):
    #     test_predict = np.array(test_predict, dtype=np.float32)
    #     test_real = np.array(test_real, dtype=np.float32)
    #     np.save(prediction_file_name, test_predict)
    #     np.save(target_file_name, test_real)
    #     np.save(output_folder + 'index_all.npy', index_all)
    #     dump_pickle(error_all, output_folder + 'arima_error.pkl')

