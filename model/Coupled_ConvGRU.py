import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell
from model.coupled_convgru_cell import Coupled_Conv2DGRUCell


class CoupledConvGRU():
    def __init__(self, input_shape=[20,10,2], input_steps=6,
                 num_layers=2, num_units=64, kernel_shape=[3,3],
                 dy_temporal=0, att_units=64,
                 dy_adj=0,
                 dy_filter=0,
                 multi_loss=0,
                 batch_size=32):
        self.input_shape = input_shape
        self.input_steps = input_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.kernel_shape = kernel_shape
        #
        self.num_nodes = np.prod(self.input_shape[:-1])
        self.dy_temporal = dy_temporal
        self.att_units = att_units
        # self.dy_adj = dy_adj
        # self.dy_filter = dy_filter
        self.multi_loss = multi_loss
        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        
        first_cell = Coupled_Conv2DGRUCell(num_units=self.num_units,
                                           input_shape=self.input_shape,
                                           kernel_shape=self.kernel_shape,
                                           num_proj=None,
                                           input_dim=self.input_shape[-1],
                                           output_dy_adj=1)
        cell = Coupled_Conv2DGRUCell(num_units=self.num_units,
                                     input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                                     kernel_shape=self.kernel_shape,
                                     num_proj=None,
                                     input_dim=self.num_units,
                                     output_dy_adj=1)
        last_cell = Coupled_Conv2DGRUCell(num_units=self.num_units,
                                          input_shape=[self.input_shape[0], self.input_shape[1], self.num_units],
                                          kernel_shape=self.kernel_shape,
                                          num_proj=None,
                                          input_dim=self.num_units,
                                          output_dy_adj=0)
        ## for only one layer
        one_cell = Coupled_Conv2DGRUCell(num_units=self.num_units,
                                         input_shape=self.input_shape,
                                         kernel_shape=self.kernel_shape,
                                         num_proj=None,
                                         input_dim=self.input_shape[-1],
                                         output_dy_adj=0)

        if num_layers > 2:
            cells = [first_cell] + [cell] * (num_layers-2) + [last_cell]
        else:
            cells = [first_cell, last_cell]
        if num_layers == 1:
            cells = [one_cell]

        self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.f = tf.placeholder(tf.float32,
                                [self.batch_size, self.input_steps, self.input_shape[0] * self.input_shape[1],
                                 self.input_shape[0] * self.input_shape[1]])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])



    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        #inputs = tf.unstack(x, axis=0)
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        # temporal attention
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        # outputs: [input_steps, batch_size, -, -, -]
        if self.multi_loss:
            if self.dy_temporal:
                with tf.variable_scope('temporal_attention', reuse=tf.AUTO_REUSE):
                    h_states = tf.transpose(outputs, (1, 0, 2, 3, 4))
                    att_states = []
                    for t in range(self.input_steps):
                        att_state, _ = self.temporal_attention_layer(outputs[t], h_states, self.att_units, reuse=tf.AUTO_REUSE)
                        att_states.append(att_state)
                    att_states = tf.stack(att_states)
                    outputs = tf.concat([outputs, att_states], -1)
            # projection
            outputs = tf.layers.dense(outputs, units=self.input_shape[-1], activation=None, kernel_initializer=self.weight_initializer)
            outputs = tf.transpose(outputs, [1, 0, 2, 3, 4])
            loss = 2 * tf.nn.l2_loss(self.y - outputs)
            return outputs, loss
        else:
            if self.dy_temporal:
                with tf.variable_scope('temporal_attention', reuse=tf.AUTO_REUSE):
                    h_states = tf.transpose(outputs[:-1], (1, 0, 2, 3, 4))
                    att_states, _ = self.temporal_attention_layer(outputs[-1], h_states, self.att_units,
                                                                  reuse=tf.AUTO_REUSE)
                    output = tf.concat([outputs[-1], att_states], -1)
            else:
                output = outputs[-1]
            # projection
            output = tf.layers.dense(output, units=self.input_shape[-1], activation=None,
                                     kernel_initializer=self.weight_initializer)
            loss = 2 * tf.nn.l2_loss(self.y[:, -1, :, :, :] - output)
            return tf.expand_dims(output, 1), loss

    '''
    # single loss
    def build_easy_model(self):
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        #inputs = tf.unstack(x, axis=0)
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.stack(outputs)
        # temporal attention
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        # outputs: [input_steps, batch_size, -, -, -]
        if self.dy_temporal:
            with tf.variable_scope('temporal_attention', reuse=tf.AUTO_REUSE):
                h_states = tf.transpose(outputs[:-1], (1,0,2,3,4))
                att_states, _ = self.temporal_attention_layer(outputs[-1], h_states, self.att_units, reuse=tf.AUTO_REUSE)
                output = tf.concat([outputs[-1], att_states], -1)
        else:
            output = outputs[-1]
        # projection
        output = tf.layers.dense(output, units=self.input_shape[-1], activation=None, kernel_initializer=self.weight_initializer)
        loss = 2 * tf.nn.l2_loss(self.y[:, -1, :, :, :] - output)
        return tf.expand_dims(output, 1), loss
    '''

    def temporal_attention_layer(self, o_state, h_states, att_units, reuse=True):
        # o_state: [batch_size, row, col, channel]
        # h_state: [batch_size, input_steps, row, col, channel]
        o_shape = o_state.get_shape().as_list()
        h_shape = h_states.get_shape().as_list()
        with tf.variable_scope('att', reuse=reuse):
            with tf.variable_scope('att_o_state', reuse=reuse):
                w = tf.get_variable('att_o_w', [np.prod(o_shape[1:]), att_units], initializer=self.weight_initializer)
                o_att = tf.matmul(tf.reshape(o_state, (-1, np.prod(o_shape[1:]))), w)
                # o_att: [batch_size, att_units]
            with tf.variable_scope('att_h_state', reuse=reuse):
                w = tf.get_variable('att_h_w', [np.prod(h_shape[2:]), att_units], initializer=self.weight_initializer)
                h_att = tf.matmul(tf.reshape(h_states, [-1, np.prod(h_shape[2:])]), w)
                # h_att: [batch_size*input_steps, att_units]
                h_att = tf.reshape(h_att, [-1, h_shape[1], att_units])
                # encoder_state_att: [batch_size, input_steps, att_units]
            b = tf.get_variable('att_b', [att_units], initializer=self.const_initializer)
            o_h_att_plus = tf.nn.relu(h_att + tf.expand_dims(o_att, 1) + b)
            mlp_w = tf.get_variable('mlp_w', [att_units, 1], initializer=self.weight_initializer)
            out_att = tf.reshape(tf.matmul(tf.reshape(o_h_att_plus, [-1, att_units]), mlp_w),
                                 [-1, h_shape[1]])
            # out_att: [batch_size, input_steps]
            alpha = tf.nn.softmax(out_att)
            context = tf.reduce_sum(tf.reshape(h_states, (-1, h_shape[1], np.prod(h_shape[2:]))) * tf.expand_dims(alpha, -1),
                                    1, name='context')
            att_context = tf.reshape(context,
                                     [-1, h_shape[2], h_shape[3], h_shape[4]])
            # att_context: [batch_size, row, col, channel]
            return att_context, alpha

    '''
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
        outputs = tf.layers.dense(tf.reshape(outputs, (-1, self.num_units)), units=self.input_shape[-1],
                                  activation=None, kernel_initializer=self.weight_initializer)
        #
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3, 4])
        loss = 2 * tf.nn.l2_loss(self.y - outputs)
        return outputs, loss
    '''








