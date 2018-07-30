from __future__ import print_function
import numpy as np
import pandas as pd
import time
import warnings
from time import mktime
from datetime import datetime
from progressbar import *

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.vector_ar.var_model import VAR
import sys
sys.path.append('../')
from utils import *
from preprocessing import *


def predict_by_samples(data, test_data, num_sample, split, output_steps):
	warnings.filterwarnings("ignore")
	index_all = np.zeros([test_data.shape[1] - output_steps, num_sample])
	valid_num = np.zeros(test_data.shape[1] - output_steps)
	error_all = []
	real = np.zeros([test_data.shape[1] - output_steps, num_sample])
	predict = np.zeros([test_data.shape[1] - output_steps, num_sample])
	#widgets = ['Train: ', Percentage(), ' ', Bar('-'), ' ', ETA()]
	#pbar = ProgressBar(widgets=widgets, maxval=test_data.shape[0]-output_steps).start()
	for t in xrange(test_data.shape[1] - output_steps):
		#pbar.update(t)
		if t%10 == 0:
			print(t)
		error_index = []
		station_sample = np.random.randint(data.shape[0], size=num_sample)
		for r in xrange(num_sample):
			# t: which time slot
			# i: which station
			i = station_sample[r]
			train_df = pd.DataFrame(data[i][t:split[0] + t])
			train_df.index = pd.DatetimeIndex(timestamps[t:split[0] + t])
			try:
				results = ARMA(train_df, order=(2, 2)).fit(trend='c', disp=-1)
			except:
				error_index.append(r)
				continue
			pre, _, _ = results.forecast(output_steps)
			test_real = test_data[i][t:t + output_steps]
			real[t, r] = test_real
			predict[t, r] = pre
		index_all[t] = station_sample
		error_all.append(error_index)
		valid_num[t] = num_sample - len(error_index)
	#pbar.finish()
	return real, predict, index_all, error_all, valid_num

def predict_by_all(data, test_data, num_sample, split, output_steps):
	warnings.filterwarnings("ignore")
	index_all = np.zeros([test_data.shape[1] - output_steps, num_sample])
	valid_num = np.zeros(test_data.shape[1] - output_steps)
	error_all = []
	real = np.zeros([test_data.shape[1] - output_steps, num_sample])
	predict = np.zeros([test_data.shape[1] - output_steps, num_sample])
	#station_sample = np.random.randint(data.shape[0], size=num_sample)
	station_sample = np.arange(data.shape[0])
	#widgets = ['Train: ', Percentage(), ' ', Bar('-'), ' ', ETA()]
	#pbar = ProgressBar(widgets=widgets, maxval=test_data.shape[0]-output_steps).start()
	for t in xrange(test_data.shape[1] - output_steps):
		#pbar.update(t)
		if t%10 == 0:
			print(t)
		error_index = []
		for r in xrange(num_sample):
			# t: which time slot
			# i: which station
			i = station_sample[r]
			train_df = pd.DataFrame(data[i][t:split[0] + t])
			train_df.index = pd.DatetimeIndex(timestamps[t:split[0] + t])
			try:
				results = ARMA(train_df, order=(2, 2)).fit(trend='c', disp=-1)
			except:
				error_index.append(r)
				continue
			pre, _, _ = results.forecast(output_steps)
			test_real = test_data[i][t:t + output_steps]
			real[t, r] = test_real
			predict[t, r] = pre
		index_all[t] = station_sample
		error_all.append(error_index)
		valid_num[t] = num_sample - len(error_index)
	#pbar.finish()
	return real, predict, index_all, error_all, valid_num

#lg_set = 1
input_steps = 6
output_steps = 1
num_sample = 50

in_save_results = False
out_save_results = False

pre_process = MinMaxNormalization01()
split = [3912, 480]
print('load train, test data...')
data, train_data, _, _ = load_npy_data(filename=['../datasets/citibike-data/data/p_station.npy', '../datasets/citibike-data/data/d_station.npy'], split=split)
# data: [num, num_station, 2]
print('preprocess train data...')
pre_process.fit(train_data)
# all_timestamps_string = gen_timestamps(['2013','2014','2015','2016'], gen_timestamps_for_year=gen_timestamps_for_year_ymdh)
# all_timestamps_string = all_timestamps_string[4344:-4416]
# all_timestamps_struct = [time.strptime(t, '%Y%m%d%H') for t in all_timestamps_string]
# timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]
all_timestamps_string = gen_timestamps(['2014'], gen_timestamps_for_year=gen_timestamps_for_year_ymdh)
# from 20140401 to 20140930
start = get_index_for_month('2014', 3)*24
end = -(get_index_for_month('2014', 12) - get_index_for_month('2014', 9))*24
all_timestamps_string = all_timestamps_string[start:end]
all_timestamps_struct = [time.strptime(t, '%Y%m%d%H') for t in all_timestamps_string]
timestamps = [datetime.fromtimestamp(mktime(t)) for t in all_timestamps_struct]

print('preprocess and get test data...')
data = pre_process.transform(data)

# data: [64*64*2, num]
#train_data = data[:][:split[0]]

#train_timestamps = timestamps[:split[0]]
# validate and test data
# val_real = np.zeros((data.shape[0], val_data.shape[-1]-output_steps, output_steps))
# val_predict = np.zeros(val_real.shape)
# test_real = np.zeros((data.shape[0], test_data.shape[-1]-output_steps, output_steps))
# test_predict = np.zeros(test_real.shape)
# ARMA for validate data
# print('======================== ARMA for validate ==========================')
# for i in range(data.shape[0]):
#     print('validate, i = '+str(i))
#     for j in range(val_data.shape[-1]-output_steps):
#         train_df = pd.DataFrame(data[i][j:split[0]+j])
#         train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+j])
#         results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
#         pre, _, _ = results.forecast(output_steps)
#         val_real[i][j] = val_data[i][j:j+output_steps]
#         val_predict[i][j] = pre
# ARMA for test data
# print('======================= ARMA for test ===============================')
# for i in range(data.shape[0]):
#     print('test, i = '+str(i))
#     for j in range(test_data.shape[-1]-output_steps):
#         train_df = pd.DataFrame(data[i][j:split[0]+split[1]+j])
#         train_df.index = pd.DatetimeIndex(timestamps[j:split[0]+split[1]+j])
#         results = ARMA(train_df, order=(2,2)).fit(trend='nc', disp=-1)
#         pre, _, _ = results.forecast(output_steps)
#         test_real[i][j] = test_data[i][j:j+output_steps]
#         test_predict[i][j] = pre
print('======================= ARMA for check-in test ===============================')
in_data = data[:, :, 0]
in_test_data = in_data[split[0]:]
in_data = np.transpose(in_data)
in_test_data = np.transpose(in_test_data)
#
if in_save_results:
	in_predict = np.load('arma_in_predict.npy')
	in_real = np.load('arma_in_real.npy')
	in_index_all = np.load('arma_in_index.npy')
	in_error_all = load_pickle('arma_in_error.pkl')
	# in_predict = pre_process.inverse_transform(in_predict)
	# in_real = pre_process.inverse_transform(in_real)
	valid_num = np.array([num_sample-len(e) for e in in_error_all])
else:
	in_real, in_predict, in_index_all, in_error_all, valid_num = predict_by_samples(in_data, in_test_data, num_sample, split, output_steps)
	in_predict = pre_process.inverse_transform(in_predict)
	in_real = pre_process.inverse_transform(in_real)
#
in_predict = np.clip(in_predict, 0, None)
in_rmse_test = np.sqrt(np.sum(np.square(in_predict-in_real))*1.0/(np.sum(valid_num)))
in_rmse_2_test = np.mean(np.sqrt(np.sum(np.square(in_predict-in_real), axis=-1)*1.0/valid_num))
in_rmlse_test = np.mean(np.sqrt(np.sum(np.square(np.log(in_predict+1)-
												 np.log(in_real+1)), axis=-1)*1.0/valid_num))
in_er_test = np.mean(np.sum(np.abs(in_predict-in_real), axis=-1)/np.sum(in_real, axis=-1))
print('test in l2-loss is %.4f' % in_rmse_test)
print('test in rmse is %.4f' % in_rmse_2_test)
print('test in rmlse is %.4f' % in_rmlse_test)
print('test in er is %.4f' % in_er_test)
np.save('arma_in_predict.npy', in_predict)
np.save('arma_in_real.npy', in_real)
np.save('arma_in_index.npy', in_index_all)
dump_pickle(in_error_all, 'arma_in_error.pkl')

print('======================= ARMA for check-out test ===============================')
out_data = data[:, :, 1]
out_test_data = out_data[split[0]:]
out_data = np.transpose(out_data)
out_test_data = np.transpose(out_test_data)
#
if out_save_results:
	out_predict = np.load('arma_in_predict.npy')
	out_real = np.load('arma_in_real.npy')
	out_index_all = np.load('arma_in_index.npy')
	out_error_all = load_pickle('arma_in_error.pkl')
	# out_predict = pre_process.inverse_transform(out_predict)
	# out_real = pre_process.inverse_transform(out_real)
	valid_num = np.array([num_sample-len(e) for e in out_error_all])
else:
	out_real, out_predict, out_index_all, out_error_all, valid_num = predict_by_samples(out_data, out_test_data, num_sample, split, output_steps)
	out_predict = pre_process.inverse_transform(out_predict)
	out_real = pre_process.inverse_transform(out_real)
#
out_predict = np.clip(out_predict, 0, None)
out_rmse_test = np.sqrt(np.sum(np.square(out_predict-out_real))*1.0/(np.sum(valid_num)))
out_rmse_2_test = np.mean(np.sqrt(np.sum(np.square(out_predict-out_real), axis=-1)*1.0/valid_num))
out_rmlse_test = np.mean(np.sqrt(np.sum(np.square(np.log(out_predict+1)-
												  np.log(out_real+1)), axis=-1)*1.0/valid_num))
out_er_test = np.mean(np.sum(np.abs(out_predict-out_real), axis=-1)/np.sum(out_real, axis=-1))
print('test out l2-loss is %.4f' % out_rmse_test)
print('test out rmse is %.4f' % out_rmse_2_test)
print('test out rmlse is %.4f' % out_rmlse_test)
print('test out er is %.4f' % out_er_test)
np.save('arma_out_predict.npy', out_predict)
np.save('arma_out_real.npy', out_real)
np.save('arma_out_index.npy', out_index_all)
dump_pickle(out_error_all, 'arma_out_error.pkl')

