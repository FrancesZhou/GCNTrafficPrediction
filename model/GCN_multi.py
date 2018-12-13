import sys
import numpy as np
import pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *
from tensorflow.contrib import legacy_seq2seq
from model.dcrnn_cell import DCGRUCell


class GCN_multi():
    def __init__(self, num_station, input_steps, output_steps,
                 num_units=64,
                 max_diffusion_step=2,
                 dy_adj=1,
                 dy_filter=0,
                 f_adj_mx=None,
                 filter_type='dual_random_walk',
                 batch_size=32):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.num_units = num_units
        self.max_diffusion_step = max_diffusion_step

        self.dy_adj = dy_adj
        self.dy_filter = dy_filter
        self.f_adj_mx = f_adj_mx
        self.filter_type = filter_type

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()


        adj_mx = self.f_adj_mx
        self.cell = DCGRUCell(self.num_units, adj_mx=adj_mx, max_diffusion_step=self.max_diffusion_step,
                              num_nodes=self.num_station, num_proj=None,
                              input_dim=self.num_station*3, dy_adj=self.dy_adj,
                              dy_filter=self.dy_filter, output_dy_adj=True,
                              reuse=tf.AUTO_REUSE, filter_type=self.filter_type)
        self.cell_with_projection = DCGRUCell(self.num_units, adj_mx=adj_mx, max_diffusion_step=max_diffusion_step,
                                              num_nodes=self.num_station, num_proj=3,
                                              input_dim=self.num_station*self.num_units, 
                                              dy_adj=self.dy_adj, dy_filter=0, output_dy_adj=False,
                                              reuse=tf.AUTO_REUSE, filter_type=self.filter_type)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 3])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.output_steps, self.num_station, 3])


    def build_model(self):
        x = tf.unstack(tf.reshape(self.x, (self.batch_size, self.input_steps, self.num_station*2)), axis=1)
        f_all = tf.unstack(tf.reshape(self.f, (self.batch_size, self.input_steps, self.num_station*self.num_station)), axis=1)
        #x = tf.unstack(tf.reshape(self.x, (-1, self.input_steps, self.num_station * 2)), axis=1)
        #f_all = tf.unstack(tf.reshape(self.f, (-1, self.input_steps, self.num_station*self.num_station)), axis=1)

        y = self.y
        hidden_state = tf.zeros([self.batch_size, self.num_station*self.num_units])
        #current_state = tf.zeros([self.batch_size, self.num_station*self.num_unists])
        #state = hidden_state, current_state
        state_1 = hidden_state
        state_2 = hidden_state
        y_ = []
        for i in range(self.input_steps):
            # for each step
            #print(i)
            f = f_all[i]
            current_step_batch = x[i]
            with tf.variable_scope('dcrnn', reuse=tf.AUTO_REUSE):
                output_1, state_1 = self.cell(tf.reshape(current_step_batch, (self.batch_size, -1)), f, state_1)
            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                output_2, state_2 = self.cell_with_projection(tf.reshape(output_1, (self.batch_size, -1)), f, state_2)
            # output: [batch_size, state_size]
            output_2 = tf.reshape(output_2, (self.batch_size, self.num_station, -1))
            y_.append(output_2)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(y-y_)
        return y_, loss


    def build_easy_model(self, is_training=False):
        encoding_cells = self.cell
        decoding_cells = self.cell_with_projection
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        # x: [input_steps, batch_size, num_station*2]
        # f_all: [input_steps, batch_size, num_station*num_station]
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        labels = tf.unstack(tf.reshape(self.y, (self.batch_size, self.input_steps, -1)), axis=1)
        #
        GO_SYMBOL = tf.zeros(shape=(self.batch_size, self.num_station * 3))

        with tf.variable_scope('GCN_SEQ'):

            labels.insert(0, GO_SYMBOL)

            # labels: (horizon+1, batch_size, num_nodes*output_dim]

            def _loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                result = tf.concat([result, f_all[-1]], axis=-1)
                #result = tf.concat([result, self.f_adj_mx], axis=-1)
                return result

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels, enc_state, decoding_cells,
                                                              loop_function=_loop_function)

        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (self.batch_size, self.input_steps, self.num_station, -1), name='outputs')
        loss = 2*tf.nn.l2_loss(self.y - self._outputs)
        #
        # self.cells = tf.contrib.rnn.MultiRNNCell([self.cell, self.cell_with_projection], state_is_tuple=True)
        # outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        # outputs = tf.stack(outputs)
        # outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.num_station, -1))
        # outputs = tf.transpose(outputs, [1, 0, 2, 3])
        # loss = 2*tf.nn.l2_loss(self.y - outputs)
        return self._outputs, loss
    
    
    
    
    
    
    
    
