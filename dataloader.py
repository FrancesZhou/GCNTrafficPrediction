from __future__ import absolute_import

import numpy as np
import math
import random
# from sklearn.model_selection import train_test_split
# import re
# import copy

class DataLoader():
    def __init__(self, d_data, f_data, e_data,
                 input_steps, output_steps,
                 num_station):
        self.d_data = d_data
        self.f_data = f_data
        self.e_data = e_data
        # d_data: [num, num_station, 2]
        # f_data: [num, {num_station, num_station}]
        # e_data: [num, 7]
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_station = num_station
        self.num_data = len(self.d_data)
        self.data_index = np.arange(self.num_data - self.input_steps)
        #self.reset_data()

    def get_flow_map_from_dict(self, f_dict):
        f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        rows = []
        cols = []
        flows = []
        for s_id, f in f_dict.items():
            ind, values = zip(*f.items())
            values = self.pre_process.transform(list(values))
            rows = rows + [s_id]*len(ind)
            cols = cols + list(ind)
            flows = flows + list(values)
        f_map[rows, cols] = flows
        return f_map

    def get_flow_map_from_list(self, f_list):
        f_map = np.zeros((self.num_station, self.num_station), dtype=np.float32)
        if len(f_list):
            rows, cols, values = zip(*f_list)
            f_map[rows, cols] = values
        return f_map

    def get_flow_indices_values_from_list(self, f_list):
        if len(f_list):
            rows, cols, values = tuple(zip(*f_list))
            return rows, cols, values
            #return np.column_stack((rows, cols)), np.array(values, dtype=np.float32)
        else:
            return None

    def next_sample(self, i):
        index = self.data_index[i]
        if index > self.num_data-self.input_steps:
            return None, None, None, None
        else:
            x = self.d_data[index: index+self.input_steps]
            #f = [self.get_flow_map_from_list(self.f_data[j]) for j in xrange(index+1, index+self.input_steps+1)]
            f = [self.get_flow_map_from_list(self.f_data[j]) for j in xrange(index, index+self.input_steps)]
            e = self.e_data[index+1: index+self.input_steps+1]
            y = self.d_data[index + 1: index + self.input_steps + 1]
            return x, f, e, y, np.arange(index+1, index+self.input_steps+1)

    def next_batch_for_train(self, start, end):
        if end > self.num_data-self.input_steps:
            return None, None, None
        else:
            # batch_x: [end-start, input_steps, num_station, 2]
            # batch_y: [end-start, input_steps, num_station, 2]
            # batch_f: [end-start, input_steps+1, num_station, num_station]
            batch_x = []
            batch_y = []
            batch_f = []
            batch_e = []
            batch_index = []
            for i in self.data_index[start:end]:
                batch_x.append(self.d_data[i: i + self.input_steps])
                batch_y.append(self.d_data[i + 1: i + self.input_steps + 1])
                f_map = [self.get_flow_map_from_list(self.f_data[j]) for j in range(i, i + self.input_steps)]
                '''
                for j in range(i, i+self.input_steps):
                    rows, cols, values = self.get_flow_indices_values_from_list(self.f_data[j])
                #f_map = [self.get_flow_indices_values_from_list(self.f_data[j]) for j in range(i, i+self.input_steps)]
                '''
                batch_f.append(f_map)
                batch_e.append(self.e_data[i+1: i+self.input_steps+1])
                batch_index.append(np.arange(i+1, i+self.input_steps+1))
            return np.array(batch_x), np.array(batch_f, dtype=np.float32), np.array(batch_e), np.array(batch_y), np.array(batch_index)

    def next_batch_for_test(self, start, end):
        padding_len = 0
        if end > self.num_data-(self.input_steps+self.output_steps-1):
            padding_len = end - (self.num_data-(self.input_steps+self.output_steps-1))
            end = self.num_data - (self.input_steps+self.output_steps-1)
        # batch_x: [end-start, input_steps, num_station, 2]
        # batch_y: [end-start, output_steps, num_station, 2]
        # batch_f: [end-start, input_steps+output_steps, num_station, num_station]
        batch_x = []
        batch_y = []
        batch_f = []
        for i in self.data_index[start:end]:
            batch_x.append(self.d_data[i: i+self.input_steps])
            batch_y.append(self.d_data[i+self.input_steps: i+self.input_steps+self.output_steps])
            #f_map = [self.get_flow_map(self.f_data[j]) for j in xrange(i+self.input_steps-1, i+self.input_steps+self.output_steps-1)]
            #batch_f.append(f_map)
        if padding_len > 0:
            batch_x = np.concatenate((np.array(batch_x), np.zeros((padding_len, self.input_steps, self.num_station, 2))), axis=0)
            batch_y = np.concatenate((np.array(batch_y), np.zeros((padding_len, self.output_steps, self.num_station, 2))), axis=0)
            # batch_f
        return batch_x, batch_y, batch_f, padding_len

    def reset_data(self):
        np.random.shuffle(self.data_index)
