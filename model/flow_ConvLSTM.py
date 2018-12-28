import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell
from model.convlstm_cell import Dy_Conv2DLSTMCell


class flow_ConvLSTM():
    def __init__(self, input_shape=[20,10,2], input_steps=6,
                 num_layers=3, num_units=32, kernel_shape=[3,3],
                 f_adj_mx=None,
                 batch_size=32):
        self.input_shape = input_shape
        self.input_steps = input_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.kernel_shape = kernel_shape

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # map - convlstm
        first_cell = Dy_Conv2DLSTMCell(input_shape=self.input_shape,
                                    output_channels=self.num_units,
                                    kernel_shape=self.kernel_shape,
                                    input_dim=self.input_shape[-1], dy_adj=0, dy_filter=0, output_dy_adj=0)
        cell = Dy_Conv2DLSTMCell(input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                              output_channels=self.num_units,
                              kernel_shape=self.kernel_shape,
                              input_dim=self.num_units, dy_adj=0, dy_filter=0, output_dy_adj=0)
        # graph - gcn
        g_first_cell = DCGRUCell(num_units=self.num_units, adj_mx=f_adj_mx, max_diffusion_step=2, num_nodes=self.input_shape[0]*self.input_shape[1],
                                 input_dim=self.input_shape[0]*self.input_shape[1]*self.input_shape[-1], dy_adj=1, dy_filter=0, output_dy_adj=1)
        g_cell = DCGRUCell(num_units=self.num_units, adj_mx=f_adj_mx, max_diffusion_step=2, num_nodes=self.input_shape[0]*self.input_shape[1],
                           input_dim=self.input_shape[0]*self.input_shape[1]*self.num_units, dy_adj=1, dy_filter=0, output_dy_adj=0)
        # concate two blocks



        if num_layers > 2:
            cells = [first_cell] + [cell] * (num_layers-2)
            g_cells = [g_first_cell] + [g_cell] * (num_layers-2)
        else:
            cells = [first_cell, cell]
            g_cells = [g_first_cell, g_cell]

        self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        self.g_cells = tf.contrib.rnn.MultiRNNCell(g_cells, state_is_tuple=True)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0]*self.input_shape[1], self.input_shape[2]])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0]*self.input_shape[1], self.input_shape[0]*self.input_shape[1]])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0]*self.input_shape[1], self.input_shape[2]])


    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], -1)), [1, 0, 2, 3, 4])
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[0]*self.input_shape[1])), [1, 0, 2, 3, 4])
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        #
        # graph convolutional network for flow information
        graph_x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        graph_f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        g_inputs = tf.concat([graph_x, graph_f_all], axis=-1)
        g_inputs = tf.unstack(g_inputs, axis=0)
        #
        g_outputs, _ = tf.contrib.rnn.static_rnn(self.g_cells, g_inputs, dtype=tf.float32)
        g_outputs = tf.stack(g_outputs)
        # combine these two blocks to output the final prediction
        outputs = tf.reshape(outputs, (-1, self.num_units))
        g_outputs = tf.reshape(g_outputs, (-1, self.num_units))
        output_concat = tf.concat([outputs, g_outputs], axis=-1)
        final_outputs = tf.layers.dense(output_concat, units=self.input_shape[-1], activation=tf.nn.relu)
        #
        final_outputs = tf.reshape(final_outputs, (self.input_steps, self.batch_size, self.input_shape[0]*self.input_shape[1], -1))
        final_outputs = tf.transpose(final_outputs, [1, 0, 2, 3])
        loss = 2 * tf.nn.l2_loss(self.y - final_outputs)
        return final_outputs, loss








