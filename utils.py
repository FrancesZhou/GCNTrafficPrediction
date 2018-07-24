import cPickle as pickle
import numpy as np
import scipy.io as sio

def dump_pickle(data, file):
    try:
        with open(file, 'w') as datafile:
            pickle.dump(data, datafile)
    except Exception as e:
        raise e

def load_pickle(file):
    try:
        with open(file, 'r') as datafile:
            data = pickle.load(datafile)
    except Exception as e:
        raise e
    return data

def load_npy_data(filename, split):
    if len(filename)==2:
        d1 = np.load(filename[0])
        d2 = np.load(filename[1])
        data = np.concatenate((np.expand_dims(d1, axis=-1), np.expand_dims(d2, axis=-1)), axis=-1)
    train = data[0:split[0]]
    validate = data[split[0]:(split[0]+split[1])]
    if len(split) > 2:
        test = data[(split[0]+split[1]):(split[0]+split[1]+split[2])]
    else:
        test = []
    return data, train, validate, test

def load_pkl_data(filename, split):
    data = load_pickle(filename)
    train = data[0:split[0]]
    validate = data[split[0]:(split[0]+split[1])]
    if len(split) > 2:
        test = data[(split[0]+split[1]):(split[0]+split[1]+split[2])]
    else:
        test = []
    return data, train, validate, test

def load_mat_data(filename, dataname, split):
    data = sio.loadmat(filename)[dataname]
    #
    max_d = np.max(data[:, -2:], axis=0)
    min_d = np.min(data[:, -2:], axis=0)
    data[:, -2:] = (data[:, -2:] - min_d)/(max_d - min_d)
    train = data[0:split[0]]
    validate = data[split[0]:(split[0]+split[1])]
    if len(split) > 2:
        test = data[(split[0]+split[1]):(split[0]+split[1]+split[2])]
    else:
        test = []
    return data, train, validate, test

def get_index_for_month(year, month):
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return np.sum(day_sum[:int(month)])

def gen_timestamps_for_year_ymd(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            t = [year+month[m]+day[d]]
            t_d = t*24
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps(years, gen_timestamps_for_year=gen_timestamps_for_year_ymd):
    timestamps = []
    for y in years:
        timestamps.append(gen_timestamps_for_year(y))
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps_for_year_ymdh(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hour1 = ['0'+str(e) for e in range(0,10)]
    hour2 = [str(e) for e in range(10,24)]
    hour = hour1+hour2
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            #t = [year+month[m]+day[d]]
            t_d = []
            for h in range(24):
                t_d.append(year+month[m]+day[d]+hour[h])
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def gen_timestamps_for_year_ymdhm(year):
    month1 = ['0'+str(e) for e in range(1,10)]
    month2 = [str(e) for e in range(10,13)]
    month = month1+month2
    day1 = ['0'+str(e) for e in range(1,10)]
    day2 = [str(e) for e in range(10,32)]
    day = day1+day2
    if year=='2012' or year=='2016':
        day_sum = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        day_sum = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hour1 = ['0'+str(e) for e in range(0,10)]
    hour2 = [str(e) for e in range(10,24)]
    hour = hour1+hour2
    #minute = ['00', '10', '20', '30', '40', '50']
    minute = ['00', '30']
    timestamps = []
    for m in range(len(month)):
        for d in range(day_sum[m]):
            #t = [year+month[m]+day[d]]
            t_d = []
            for h in range(24):
                a = [year+month[m]+day[d]+hour[h]+e for e in minute]
                #t_d = [t_d.append(year+month[m]+day[d]+hour[h]+e) for e in minute]
                t_d.append(a)
            t_d = np.hstack(np.array(t_d))
            timestamps.append(t_d[:])
    timestamps = np.hstack(np.array(timestamps))
    return timestamps

def batch_data(d_data, f_data, batch_size=32, input_steps=10, output_steps=10):
    # d_data: [num, num_station, 2]
    # f_data: [num, {num_station, num_station}]
    num = d_data.shape[0]
    assert len(f_data) == num
    # x: [batches, batch_size, input_steps, num_station, 2]
    # y: [batches, batch_size, output_steps, num_station, 2]
    # f: [batches, batch_size, input_steps+output_steps, {num_station, num_station}]
    x = []
    y = []
    f = []
    i = 0
    while i<num-batch_size-input_steps-output_steps:
        batch_x = []
        batch_y = []
        batch_f = []
        for s in range(batch_size):
            batch_x.append(d_data[(i+s): (i+s+input_steps)])
            batch_y.append(d_data[(i+s+input_steps): (i+s+input_steps+output_steps)])
            batch_f.append(f_data[(i+s): (i+s+input_steps+output_steps)])
        x.append(batch_x)
        y.append(batch_y)
        f.append(batch_f)
        i += batch_size
    return x, y, f

def get_embedding_from_file(file, num):
    with open(file, 'r') as df:
        lines = df.readlines()
        _, dim = lines[0].split(' ', 1)
        #num = int(num)
        dim = int(dim)
        embeddings = np.zeros((num, dim), dtype=np.float32)
        for line in lines[1:]:
            label, v_str = line.split(' ', 1)
            v = [float(e) for e in v_str.split()]
            embeddings[int(label)] = v
    return embeddings

def get_loss(y, y_out):
    # y, y_out: [num_station, 2]
    # check-in loss
    y = np.transpose(y)
    y_out = np.transpose(y_out)
    in_rmse = np.sqrt(np.sum(np.square(y_out[0]-y[0])))
    out_rmse = np.sqrt(np.sum(np.square(y_out[1]-y[1])))
    in_rmlse = np.sqrt(np.mean(np.square(np.log(y_out[0] + 1)-np.log(y[0] + 1))))
    out_rmlse = np.sqrt(np.mean(np.square(np.log(y_out[1] + 1)-np.log(y[1] + 1))))
    in_sum = np.max((np.sum(y[0]), 1))
    out_sum = np.max((np.sum(y[1]), 1))
    #print in_sum.shape
    in_er = np.sum(np.abs(y_out[0]-y[0]))/in_sum
    out_er = np.sum(np.abs(y_out[1]-y[1]))/out_sum
    return [in_rmse, out_rmse, in_rmlse, out_rmlse, in_er, out_er]
