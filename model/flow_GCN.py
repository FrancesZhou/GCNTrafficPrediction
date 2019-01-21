import sys
import numpy as np
import pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell


class flow_GCN():
    def __init__(self, num_station, input_steps,
                 num_layers=2, num_units=64,
                 max_diffusion_step=2,
                 f_adj_mx=None,
                 filter_type='dual_random_walk',
                 batch_size=32):
        self.num_station = num_station
        self.input_steps = input_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.max_diffusion_step = max_diffusion_step

        self.f_adj_mx = f_adj_mx
        self.filter_type = filter_type

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()


        adj_mx = self.f_adj_mx
        first_cell = DCGRUCell(self.num_units, adj_mx=adj_mx, max_diffusion_step=self.max_diffusion_step,
                               num_nodes=self.num_station, num_proj=None,
                               input_dim=2,
                               dy_adj=0, dy_filter=0, output_dy_adj=0)
        cell = DCGRUCell(self.num_units, adj_mx=adj_mx, max_diffusion_step=self.max_diffusion_step,
                         num_nodes=self.num_station, num_proj=None,
                         input_dim=self.num_units,
                         dy_adj=0, dy_filter=0, output_dy_adj=0)
        ######
        f_first_cell = DCGRUCell(self.num_units, adj_mx=None, max_diffusion_step=self.max_diffusion_step,
                                 num_nodes=self.num_station, num_proj=None,
                                 input_dim=2,
                                 dy_adj=1, dy_filter=0, output_dy_adj=1)
        f_cell = DCGRUCell(self.num_units, adj_mx=None, max_diffusion_step=self.max_diffusion_step,
                           num_nodes=self.num_station, num_proj=None,
                           input_dim=self.num_units,
                           dy_adj=1, dy_filter=0, output_dy_adj=1)
        f_last_cell = DCGRUCell(self.num_units, adj_mx=None, max_diffusion_step=self.max_diffusion_step,
                                num_nodes=self.num_station, num_proj=None,
                                input_dim=self.num_units,
                                dy_adj=1, dy_filter=0, output_dy_adj=0)
        if num_layers>2:
            cells = [first_cell] + [cell]*(num_layers-1)
            f_cells = [f_first_cell] + [f_cell]*(num_layers-2) + [f_last_cell]
        else:
            cells = [first_cell, cell]
            f_cells = [f_first_cell, f_last_cell]

        self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        self.f_cells = tf.contrib.rnn.MultiRNNCell(f_cells, state_is_tuple=True)


        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])



    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        # x: [input_steps, batch_size, num_station*2]
        # f_all: [input_steps, batch_size, num_station*num_station]
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        #
        f_outputs, _ = tf.contrib.rnn.static_rnn(self.f_cells, inputs, dtype=tf.float32)
        f_outputs = tf.stack(f_outputs)
        #
        # combine these two blocks to output the final prediction
        outputs = tf.reshape(outputs, (-1, self.num_units))
        f_outputs = tf.reshape(f_outputs, (-1, self.num_units))
        output_concat = tf.concat([outputs, f_outputs], axis=-1)
        final_outputs = tf.layers.dense(output_concat, units=2, activation=tf.nn.relu, kernel_initializer=self.weight_initializer)
        #
        final_outputs = tf.reshape(final_outputs, (self.input_steps, self.batch_size, self.num_station, -1))
        final_outputs = tf.transpose(final_outputs, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(self.y - final_outputs)
        return final_outputs, loss
    
    
    
    
    
    
    
    
