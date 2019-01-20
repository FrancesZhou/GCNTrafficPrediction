import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell
from model.convlstm_cell import Dy_Conv2DLSTMCell
from model.coupled_convgru_cell import Coupled_Conv2DGRUCell


class CoupledConvGRU():
    def __init__(self, input_shape=[20,10,2], input_steps=6,
                 num_layers=3, num_units=32, kernel_shape=[3,3],
                 dy_adj=0,
                 dy_filter=0,
                 batch_size=32):
        self.input_shape = input_shape
        self.input_steps = input_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.kernel_shape = kernel_shape
        # self.dy_adj = dy_adj
        # self.dy_filter = dy_filter

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        
#         if self.dy_adj > 0:
#             output_dy_adj = True
#         else:
#             output_dy_adj = False
        
        first_cell = Coupled_Conv2DGRUCell(num_units=self.num_units, input_shape=self.input_shape,
                                           kernel_shape=self.kernel_shape,
                                           num_proj=None,
                                           input_dim=self.input_shape[-1], output_dy_adj=1)
        cell = Coupled_Conv2DGRUCell(num_units=self.num_units, input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                                     kernel_shape=self.kernel_shape,
                                     num_proj=None,
                                     input_dim=self.num_units, output_dy_adj=1)
        last_cell = Coupled_Conv2DGRUCell(num_units=self.input_shape[-1], input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                                          kernel_shape=self.kernel_shape,
                                          num_proj=None,
                                          input_dim=self.num_units, output_dy_adj=0)

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
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, self.input_shape[0]*self.input_shape[1], -1)), [1, 0, 2, 3])
        #inputs = tf.unstack(x, axis=0)
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, self.input_shape[0]*self.input_shape[1], self.input_shape[0] * self.input_shape[1])), [1, 0, 2, 3])
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3, 4])
        loss = 2 * tf.nn.l2_loss(self.y - outputs)
        return outputs, loss








