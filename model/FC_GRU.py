import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
sys.path.append('./util/')
from utils import *


class FC_GRU():
    def __init__(self, num_station, input_steps,
                 num_layers=2, num_units=64,
                 max_diffusion_step=2,
                 dy_adj=1,
                 dy_filter=0,
                 f_adj_mx=None,
                 filter_type='dual_random_walk',
                 batch_size=32):
        self.num_station = num_station
        self.input_steps = input_steps
        self.num_units = num_units

        # self.max_diffusion_step = max_diffusion_step
        # self.dy_adj = dy_adj
        # self.dy_filter = dy_filter
        # self.f_adj_mx = f_adj_mx
        # self.filter_type = filter_type

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()


        cells = [tf.contrib.rnn.GRUCell(self.num_units, forget_bias=1.0, name='lstm_{0}'.format(i)) for i in range(num_layers)]
        # cell_with_projection = tf.contrib.rnn.BasicLSTMCell(self.num_station*2, forget_bias=1.0)
        # cells = [cell] * num_layers
        self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)


        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])



    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        #f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        #inputs = tf.concat([x, f_all], axis=-1)
        #inputs = tf.unstack(inputs, axis=0)
        inputs = tf.unstack(x, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        # projection
        outputs = tf.layers.dense(tf.reshape(outputs, (-1, self.num_units)), units=self.num_station*2, activation=None, kernel_initializer=self.weight_initializer)
        #
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.num_station, -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3])
        # outputs = outputs + self.x
        loss = 2*tf.nn.l2_loss(self.y - outputs)
        return outputs, loss
    
    
    
    
    
    
    
    
