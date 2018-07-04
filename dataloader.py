from __future__ import absolute_import

import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
# import re
# import copy

class DataLoader():
    def __init__(self, d_data, f_data,
                 input_steps, output_steps,
                 num_station,
                 pre_process):
        self.d_data = d_data
        self.f_data = f_data
        # d_data: [num, num_station, 2]
        # f_data: [num, {num_station, num_station}]
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_station = num_station
        self.pre_process = pre_process
        self.num_data = len(self.d_data)
        self.data_index = np.arange(self.num_data)
        self.reset_data()
    # def initialize_dataloader(self):
    #     print 'num of doc:             ' + str(len(self.doc_wordID_data))
    #     print 'num of y:               ' + str(len(self.label_data))
    #     print 'num of candidate_label: ' + str(len(self.candidate_label_data))
    #
    #     print 'after removing zero-length data'
    #     print 'num of doc:             ' + str(len(self.doc_wordID_data))
    #     print 'num of y:               ' + str(len(self.label_data))
    #     print 'num of candidate_label: ' + str(len(self.candidate_label_data))
    #     self.reset_data()

    def get_flow_map(self, f_dict):
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

    def next_batch_for_train(self, start, end):
        end = min(self.num_data, end)
        # batch_x: [end-start, input_steps, num_station, 2]
        # batch_y: [end-start, input_steps, num_station, 2]
        # batch_f: [end-start, input_steps+1, num_station, num_station]
        batch_x = []
        batch_y = []
        batch_f = []
        for i in self.data_index[start:end]:
            batch_x.append(self.d_data[i: i + self.input_steps])
            batch_y.append(self.d_data[i + 1: i + self.input_steps + 1])
            f_map = [self.get_flow_map(self.f_data[j]) for j in xrange(i, i + self.input_steps)]
            batch_f.append(f_map)
        return batch_x, batch_y, batch_f

    def next_batch_for_test(self, start, end):
        end = min(self.num_data, end)
        # batch_x: [end-start, input_steps, num_station, 2]
        # batch_y: [end-start, output_steps, num_station, 2]
        # batch_f: [end-start, input_steps+output_steps, num_station, num_station]
        batch_x = []
        batch_y = []
        batch_f = []
        for i in self.data_index[start:end]:
            batch_x.append(self.d_data[i: i+self.input_steps])
            batch_y.append(self.d_data[i+self.input_steps: i+self.input_steps+self.output_steps])
            f_map = [self.get_flow_map(self.f_data[j]) for j in xrange(i+self.input_steps-1, i+self.input_steps+self.output_steps-1)]
            batch_f.append(f_map)
        return batch_x, batch_y, batch_f

    def reset_data(self):
        np.random.shuffle(self.data_index)
