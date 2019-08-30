import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell
from model.convlstm_cell import Dy_Conv2DLSTMCell


class ConvLSTM():
    def __init__(self, input_shape=[20,10,2], input_steps=6,
                 num_layers=3, num_units=64, kernel_shape=[3,3],
                 dy_adj=0,
                 dy_filter=0,
                 batch_size=32):
        self.input_shape = input_shape
        self.input_steps = input_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.kernel_shape = kernel_shape
        self.dy_adj = dy_adj
        self.dy_filter = dy_filter

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        
#         if self.dy_adj > 0:
#             output_dy_adj = True
#         else:
#             output_dy_adj = False
        
        first_cell = Dy_Conv2DLSTMCell(input_shape=self.input_shape,
                                    output_channels=self.num_units,
                                    kernel_shape=self.kernel_shape,
                                    input_dim=self.input_shape[-1], dy_adj=self.dy_adj, dy_filter=self.dy_filter, output_dy_adj=self.dy_adj)
        cell = Dy_Conv2DLSTMCell(input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                              output_channels=self.num_units,
                              kernel_shape=self.kernel_shape,
                              input_dim=self.num_units, dy_adj=self.dy_adj, dy_filter=self.dy_filter, output_dy_adj=self.dy_adj)
        last_cell = Dy_Conv2DLSTMCell(input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                                   output_channels=self.input_shape[-1],
                                   kernel_shape=self.kernel_shape,
                                   input_dim=self.num_units, dy_adj=self.dy_adj, dy_filter=self.dy_filter, output_dy_adj=0)

        if num_layers > 2:
            cells = [first_cell] + [cell] * (num_layers-2) + [last_cell]
        else:
            cells = [first_cell, last_cell]

        self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.f = tf.placeholder(tf.float32,
                                [self.batch_size, self.input_steps, self.input_shape[0] * self.input_shape[1],
                                 self.input_shape[0] * self.input_shape[1]])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])


    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], -1)), [1, 0, 2, 3, 4])
        inputs = tf.unstack(x, axis=0)
        #f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[0] * self.input_shape[1])), [1, 0, 2, 3, 4])
        #inputs = tf.concat([x, f_all], axis=-1)
        #inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        #
        # projection
        outputs = tf.layers.dense(tf.reshape(outputs, (-1, self.num_units)), units=self.input_shape[-1],
                                  activation=None, kernel_initializer=self.weight_initializer)
        #
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3, 4])
        loss = 2 * tf.nn.l2_loss(self.y - outputs)
        return outputs, loss








