import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell
from model.coupled_convgru_cell import Coupled_Conv2DGRUCell
from model.coupled_dcrnn_cell import Coupled_DCGRUCell


class Coupled_GCN():
    def __init__(self, num_station, input_steps=6,
                 num_layers=2, num_units=64,
                 max_diffusion_step=2,
                 dy_adj=0,
                 dy_filter=0,
                 f_adj_mx=None,
                 filter_type='dual_random_walk',
                 batch_size=32):
        self.num_nodes = num_station
        self.input_steps = input_steps
        # self.num_layers = num_layers
        self.num_units = num_units
        #
        self.max_diffusion_steps = max_diffusion_step
        self.f_adj_mx = f_adj_mx
        self.filter_type = filter_type
        #
        # self.dy_adj = dy_adj
        # self.dy_filter = dy_filter

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        #
        
        first_cell = Coupled_DCGRUCell(num_units=self.num_units, adj_mx=f_adj_mx,
                                       max_diffusion_step=self.max_diffusion_steps,
                                       num_nodes=self.num_nodes, num_proj=None,
                                       input_dim=2,
                                       output_dy_adj=1)
        cell = Coupled_DCGRUCell(num_units=self.num_units, adj_mx=f_adj_mx,
                                 max_diffusion_step=self.max_diffusion_steps,
                                 num_nodes=self.num_nodes, num_proj=None,
                                 input_dim=self.num_units,
                                 output_dy_adj=1)
        last_cell = Coupled_DCGRUCell(num_units=self.num_nodes, adj_mx=f_adj_mx,
                                      max_diffusion_step=self.max_diffusion_steps,
                                      num_nodes=self.num_nodes, num_proj=None,
                                      input_dim=self.num_units,
                                      output_dy_adj=0)

        if num_layers > 2:
            cells = [first_cell] + [cell] * (num_layers-2) + [last_cell]
        else:
            cells = [first_cell, last_cell]
        #cells = [first_cell, last_cell]

        self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_nodes, 2])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_nodes, self.num_nodes])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_nodes, 2])


    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        #inputs = tf.unstack(x, axis=0)
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        #
        # projection
        outputs = tf.layers.dense(tf.reshape(outputs, (-1, self.num_units)), units=2,
                                  activation=tf.nn.relu, kernel_initializer=self.weight_initializer)
        #
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.num_nodes, -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3])
        loss = 2 * tf.nn.l2_loss(self.y - outputs)
        return outputs, loss








