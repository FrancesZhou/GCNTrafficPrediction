from __future__ import absolute_import

import numpy as np
from scipy.sparse import csr_matrix
import math
import random
# from sklearn.model_selection import train_test_split
# import re
# import copy

class DataLoader_graph():
    def __init__(self, d_data, f_data,
                 input_steps,
                 flow_format='identity'):
        self.d_data = d_data
        self.f_data = f_data
        # d_data: [num, num_station, 2]
        # f_data: [num, {num_station, num_station}]
        self.input_steps = input_steps
        self.num_station = np.prod(self.d_data.shape[1:-1])
        #self.d_data_shape = self.d_data.shape[1:]
        self.num_data = len(self.d_data)
        self.data_index = np.arange(self.num_data - self.input_steps)
        if flow_format == 'rowcol':
            self.get_flow_map_from_list = self.get_flow_map_from_list_rowcol
        elif flow_format == 'index':
            self.get_flow_map_from_list = self.get_flow_map_from_list_index
        elif flow_format == 'identity':
            self.get_flow_map_from_list = self.get_flow_map_from_identity
        #self.reset_data()

    def get_flow_adj_mx(self):
        f_adj_mx = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        for i in range(len(self.f_data)):
            #f_list = self.f_data[i]
            f_map = self.get_flow_map_from_list(self.f_data[i])
#             f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
#             if len(f_list):
#                 rows, cols, values = zip(*f_list)
#                 f_map[rows, cols] = values
            f_adj_mx = f_adj_mx + f_map
        return f_adj_mx

    def _num_batches(self, batch_size, use_all_data=False):
        if use_all_data:
            #print(self.num_data)
            #print((self.num_data - self.input_steps - self.output_steps + 1)/batch_size)
            return math.ceil((self.num_data - self.input_steps)/batch_size)
        else:
            return (self.num_data - self.input_steps)//batch_size
    
    def get_flow_map_from_list_index(self, f_list):
        f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        data, indices, indptr = f_list
        if len(data):
            f_map = csr_matrix((data, indices, indptr), shape=(self.num_station, self.num_station), dtype=np.float32).toarray()
        return f_map

    def get_flow_map_from_list_rowcol(self, f_list):
        f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        if len(f_list):
            rows, cols, values = zip(*f_list)
            f_map[rows, cols] = values
        return f_map

    def get_flow_map_from_identity(self, f_list):
        return f_list


    def next_batch_for_train(self, start, end):
        if end > self.num_data-self.input_steps:
            return None
        else:
            # batch_x: [end-start, input_steps, num_station, 2]
            # batch_y: [end-start, input_steps, num_station, 2]
            # batch_f: [end-start, input_steps, num_station, num_station]
            batch_x = []
            batch_y = []
            batch_f = []
            batch_index = []
            for i in self.data_index[start:end]:
                batch_x.append(self.d_data[i: i + self.input_steps])
                batch_y.append(self.d_data[i + 1: i + self.input_steps + 1])
                f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
                batch_f.append(f_map)
                batch_index.append(np.arange(i+1, i+self.input_steps+1))
            return np.array(batch_x,dtype=np.float32), np.array(batch_f,dtype=np.float32), np.array(batch_y), np.array(batch_index)

    def next_batch_for_test(self, start, end):
        padding_len = 0
        if end > self.num_data-self.input_steps:
            padding_len = end - (self.num_data-self.input_steps)
            end = self.num_data - self.input_steps
        # batch_x: [end-start, input_steps, num_station, 2]
        # batch_y: [end-start, output_steps, num_station, 2]
        # batch_f: [end-start, input_steps, num_station, num_station]
        batch_x = []
        batch_y = []
        batch_f = []
        batch_index = []
        for i in self.data_index[start:end]:
            batch_x.append(self.d_data[i: i+self.input_steps])
            batch_y.append(self.d_data[i+1: i+self.input_steps+1])
            f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
            batch_f.append(f_map)
            batch_index.append(np.arange(i + 1, i + self.input_steps + 1))
        if padding_len > 0:
            batch_x = np.concatenate((np.array(batch_x), np.zeros((padding_len, self.input_steps, self.num_station, 2))), axis=0)
            batch_y = np.concatenate((np.array(batch_y), np.zeros((padding_len, self.input_steps, self.num_station, 2))), axis=0)
            batch_f = np.concatenate((np.array(batch_f), np.zeros((padding_len, self.input_steps, self.num_station, self.num_station))), axis=0)
            batch_index = np.concatenate((np.array(batch_index), np.zeros((padding_len, self.input_steps))), axis=0)
        return np.array(batch_x,dtype=np.float32), np.array(batch_f,dtype=np.float32), np.array(batch_y), np.array(batch_index, dtype=np.int32), padding_len

    def reset_data(self):
        np.random.shuffle(self.data_index)


class DataLoader_map():
    def __init__(self, d_data, f_data,
                 input_steps,
                 flow_format='identity'):
        self.d_data = d_data
        self.f_data = f_data
        # d_data: [num, height, width, 2]
        # f_data: [num, height*width, height*width]
        self.input_steps = input_steps
        #
        d_data_shape = self.d_data.shape
        self.map_size = d_data_shape[1:-1]
        self.input_dim = d_data_shape[-1]
        #self.d_data_shape = d_data_shape[1:]
        self.num_data = d_data_shape[0]
        #
        self.f_data_shape = self.f_data.shape
        self.data_index = np.arange(self.num_data - self.input_steps)
        if flow_format == 'rowcol':
            self.get_flow_map_from_list = self.get_flow_map_from_list_rowcol
        elif flow_format == 'index':
            self.get_flow_map_from_list = self.get_flow_map_from_list_index
        elif flow_format == 'identity':
            self.get_flow_map_from_list = self.get_flow_map_identity
        #self.reset_data()

    def get_flow_map_from_list_index(self, f_list, f_shape):
        f_map = np.zeros(f_shape, dtype=np.float32)
        data, indices, indptr = f_list
        if len(data):
            f_map = csr_matrix((data, indices, indptr), shape=f_shape, dtype=np.float32).toarray()
        return f_map

    def get_flow_map_from_list_rowcol(self, f_list, f_shape):
        f_map = np.zeros(f_shape, dtype=np.float32)
        if len(f_list):
            rows, cols, values = zip(*f_list)
            f_map[rows, cols] = values
        return f_map

    def get_flow_map_identity(self, f_list):
        return f_list

    def _num_batches(self, batch_size, use_all_data=False):
        if use_all_data:
            #print(self.num_data)
            #print((self.num_data - self.input_steps - self.output_steps + 1)/batch_size)
            return math.ceil((self.num_data - self.input_steps)/batch_size)
        else:
            return (self.num_data - self.input_steps)//batch_size
    
    def next_batch_for_train(self, start, end):
        if end > self.num_data-self.input_steps:
            return None
        else:
            # batch_x: [end-start, input_steps, h,w, 2]
            # batch_y: [end-start, input_steps, h,w, 2]
            # batch_f: [end-start, input_steps, h,w, nb_size*nb_size]
            batch_x = []
            batch_y = []
            batch_f = []
            batch_index = []
            for i in self.data_index[start:end]:
                batch_x.append(self.d_data[i: i + self.input_steps])
                batch_y.append(self.d_data[i + 1: i + self.input_steps + 1])
                f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
                batch_f.append(f_map)
                batch_index.append(np.arange(i+1, i+self.input_steps+1))
            return np.array(batch_x), np.array(batch_f, dtype=np.float32), np.array(batch_y), np.array(batch_index)

    def next_batch_for_test(self, start, end):
        padding_len = 0
        if end > self.num_data-self.input_steps:
            padding_len = end - (self.num_data-self.input_steps)
            end = self.num_data - self.input_steps
        # batch_x: [end-start, input_steps, h,w, 2]
        # batch_y: [end-start, output_steps, h,w, 2]
        # batch_f: [end-start, input_steps, h,w, nb_size*nb_size]
        batch_x = []
        batch_y = []
        batch_f = []
        batch_index = []
        for i in self.data_index[start:end]:
            batch_x.append(self.d_data[i: i+self.input_steps])
            batch_y.append(self.d_data[i+1: i+self.input_steps+1])
            f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
            batch_f.append(f_map)
            batch_index.append(np.arange(i + 1, i + self.input_steps + 1))
        if padding_len > 0:
            batch_x = np.concatenate((np.array(batch_x), np.zeros((padding_len, self.input_steps, self.map_size[0], self.map_size[1], self.input_dim))), axis=0)
            batch_y = np.concatenate((np.array(batch_y), np.zeros((padding_len, self.input_steps, self.map_size[0], self.map_size[1], self.input_dim))), axis=0)
            batch_f = np.concatenate((np.array(batch_f), np.zeros((padding_len, self.input_steps, self.f_data_shape[1], self.f_data_shape[-1]))), axis=0)
            batch_index = np.concatenate((np.array(batch_index), np.zeros((padding_len, self.input_steps))), axis=0)
        return np.array(batch_x), np.array(batch_f), np.array(batch_y), np.array(batch_index, dtype=np.int32), padding_len

    def reset_data(self):
        np.random.shuffle(self.data_index)


class DataLoader_multi_graph():
    def __init__(self, d_data, f_data, input_dim,
                 input_steps,
                 output_steps,
                 num_station,
                 flow_format='identity'):
        self.d_data = d_data
        self.f_data = f_data
        self.input_dim = input_dim
        #
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_station = num_station
        self.num_data = len(self.d_data)
        self.data_index = np.arange(self.num_data - self.input_steps - self.output_steps + 1)
        if flow_format == 'rowcol':
            self.get_flow_map_from_list = self.get_flow_map_from_list_rowcol
        elif flow_format == 'index':
            self.get_flow_map_from_list = self.get_flow_map_from_list_index
        elif flow_format == 'identity':
            self.get_flow_map_from_list = self.get_flow_map_from_identity
        #self.reset_data()

    def get_flow_adj_mx(self):
        f_adj_mx = np.mean(self.f_data, axis=0)
        return f_adj_mx

    def get_flow_map_from_list_index(self, f_list):
        f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        data, indices, indptr = f_list
        if len(data):
            f_map = csr_matrix((data, indices, indptr), shape=(self.num_station, self.num_station), dtype=np.float32).toarray()
        return f_map

    def get_flow_map_from_list_rowcol(self, f_list):
        f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        if len(f_list):
            rows, cols, values = zip(*f_list)
            f_map[rows, cols] = values
        return f_map

    def get_flow_map_from_identity(self, f_list):
        return f_list

    def generate_graph_seq2seq_io_data(self):
        x_offsets = np.sort(np.concatenate((np.arange(1-self.input_steps, 1, 1),)))
        y_offsets = np.sort(np.arange(1, self.output_steps+1, 1))
        x, y = [], []
        # t is the index of the last observation.
        min_t = abs(min(x_offsets))
        max_t = abs(self.num_data - abs(max(y_offsets)))  # Exclusive
        for t in range(min_t, max_t):
            x_t = self.d_data[t + x_offsets, ...]
            y_t = self.d_data[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)

    def _num_batches(self, batch_size, use_all_data=False):
        if use_all_data:
            print(self.num_data)
            print((self.num_data - self.input_steps - self.output_steps + 1)/batch_size)
            return math.ceil((self.num_data - self.input_steps - self.output_steps + 1)/batch_size)
        else:
            return (self.num_data - self.input_steps -self.output_steps + 1)//batch_size

    def next_batch_for_train(self, start, end):
        if end > self.num_data-self.output_steps-self.input_steps+1:
            return None
        else:
            # batch_x: [end-start, input_steps, num_station, 2]
            # batch_y: [end-start, input_steps, num_station, 2]
            # batch_f: [end-start, input_steps, num_station, num_station]
            batch_x = []
            batch_y = []
            batch_f = []
            batch_index = []
            for i in self.data_index[start:end]:
                batch_x.append(self.d_data[i: i + self.input_steps])
                batch_y.append(self.d_data[i + self.input_steps: i + self.input_steps + self.output_steps])
                f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
                batch_f.append(f_map)
                batch_index.append(np.arange(i+self.input_steps, i+self.input_steps+self.output_steps))
            return np.array(batch_x), np.array(batch_f, dtype=np.float32), np.array(batch_y), np.array(batch_index)

    def next_batch_for_test(self, start, end):
        padding_len = 0
        if end > self.num_data-self.input_steps-self.output_steps+1:
            padding_len = end - (self.num_data-self.input_steps-self.output_steps+1)
            end = self.num_data - self.input_steps - self.output_steps + 1
        # batch_x: [end-start, input_steps, num_station, 2]
        # batch_y: [end-start, output_steps, num_station, 2]
        # batch_f: [end-start, input_steps, num_station, num_station]
        batch_x = []
        batch_y = []
        batch_f = []
        batch_index = []
        for i in self.data_index[start:end]:
            batch_x.append(self.d_data[i: i+self.input_steps])
            batch_y.append(self.d_data[i+self.input_steps: i+self.input_steps+self.output_steps])
            f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
            batch_f.append(f_map)
            batch_index.append(np.arange(i + self.input_steps, i + self.input_steps + self.output_steps))
        if padding_len > 0:
            batch_x = np.concatenate((np.array(batch_x), np.zeros((padding_len, self.input_steps, self.num_station, self.input_dim))), axis=0)
            batch_y = np.concatenate((np.array(batch_y), np.zeros((padding_len, self.output_steps, self.num_station, self.input_dim))), axis=0)
            batch_f = np.concatenate((np.array(batch_f), np.zeros((padding_len, self.input_steps, self.num_station, self.num_station))), axis=0)
        return np.array(batch_x), np.array(batch_f, dtype=np.float32), np.array(batch_y), np.array(batch_index), padding_len

    def next_batch_for_final_test(self, batch_size):
        padding_len = batch_size - 1
        # batch_x: [end-start, input_steps, num_station, 2]
        # batch_y: [end-start, output_steps, num_station, 2]
        # batch_f: [end-start, input_steps, num_station, num_station]
        batch_x = np.expand_dims(self.d_data[-self.input_steps:], axis=0)
        # give zero fake y
        batch_y = np.zeros((1, self.output_steps, self.num_station, self.input_dim))
        batch_f = np.expand_dims([self.get_flow_map_from_list(self.f_data[j]) for j in range(-self.input_steps, 0)], axis=0)
        batch_index = [np.arange(self.num_data, self.num_data+self.output_steps)]
        if padding_len > 0:
            batch_x = np.concatenate((np.array(batch_x), np.zeros((padding_len, self.input_steps, self.num_station, self.input_dim))), axis=0)
            batch_y = np.concatenate((np.array(batch_y), np.zeros((padding_len, self.output_steps, self.num_station, self.input_dim))), axis=0)
            batch_f = np.concatenate((np.array(batch_f), np.zeros((padding_len, self.input_steps, self.num_station, self.num_station))), axis=0)
        return np.array(batch_x), np.array(batch_f, dtype=np.float32), np.array(batch_y), np.array(batch_index), padding_len

    def reset_data(self):
        np.random.shuffle(self.data_index)
