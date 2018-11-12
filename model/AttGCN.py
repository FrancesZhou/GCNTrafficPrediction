import sys
import numpy as np
import pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell


class AttGCN():
    def __init__(self, num_station, input_steps, output_steps,
                 ext_dim=7,
                 num_units=64,
                 max_diffusion_step=2,
                 dy_adj=1,
                 f_adj_mx=None,
                 filter_type='dual_random_walk',
                 batch_size=32,
                 add_ext=0,
                 att_dy_adj=1):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.ext_dim = ext_dim
        self.num_units = num_units
        self.max_diffusion_step = max_diffusion_step
        self.f_adj_mx = f_adj_mx
        self.filter_type = filter_type

        self.batch_size = batch_size
        self.add_ext = add_ext
        self.att_dy_adj=att_dy_adj

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0, dtype=tf.float32)
        self.neg_inf = tf.constant(value=-np.inf, name='numpy_neg_inf')
        self.output_dim = 2

        if dy_adj:
            self.cell = DCGRUCell(self.num_units, self.max_diffusion_step, self.num_station, adj_mx=None, reuse=tf.AUTO_REUSE, filter_type=self.filter_type)
        else:
            self.cell = DCGRUCell(self.num_units, self.max_diffusion_step, self.num_station, adj_mx=self.f_adj_mx,
                                  reuse=tf.AUTO_REUSE, filter_type=self.filter_type)
        #self.cell_with_projection = DCGRUCell(self.num_units, max_diffusion_step=max_diffusion_step, num_nodes=self.num_station,  adj_mx=self.f_adj_mx, num_proj=2, filter_type=self.filter_type)


        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.e = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.ext_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])


    def build_model(self):
        x = tf.unstack(tf.reshape(self.x, (self.batch_size, self.input_steps, self.num_station*2)), axis=1)
        f_all = tf.unstack(tf.reshape(self.f, (self.batch_size, self.input_steps, self.num_station*self.num_station)), axis=1)

        e_all = tf.transpose(self.e, [1, 0, 2])
        adj_mx = tf.manip.tile(tf.expand_dims(self.f_adj_mx, 0), (self.batch_size, 1, 1))
        y = self.y
        hidden_state = tf.zeros([self.batch_size, self.num_station*self.num_units])
        #current_state = tf.zeros([self.batch_size, self.num_station*self.num_unists])
        #state = hidden_state, current_state
        state_1 = hidden_state
        #state_2 = hidden_state
        y_ = []
        for i in range(self.input_steps):
            # for each step
            #print(i)
            f = f_all[i]
            #current_step_batch = x[i]
            output_1 = x[i]
            with tf.variable_scope('dcrnn', reuse=tf.AUTO_REUSE):
                output_1, state_1 = self.cell(tf.reshape(output_1, (self.batch_size, -1)), f, state_1)
            # with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            #     output_2, state_2 = self.cell_with_projection(tf.reshape(output_1, (self.batch_size, -1)), f, state_2)
            # output: [batch_size, state_size]
            # output_2 = tf.reshape(output_2, (self.batch_size, self.num_station, -1))

            #
            with tf.variable_scope('attention', reuse=tf.AUTO_REUSE):
                output_1 = tf.reshape(output_1, (self.batch_size, self.num_station, -1))
                if self.att_dy_adj:
                    att_1, att_2 = self.att_head(output_1, adj_mx=f)
                else:
                    att_1, att_2 = self.att_head(output_1, adj_mx=adj_mx)
            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                cat_input = tf.concat([output_1, att_1, att_2], axis=-1)
                cat_dim = cat_input.get_shape()[-1].value
                cat_input = tf.reshape(cat_input, (-1, cat_dim))
                w = tf.get_variable('w', shape=[cat_dim, self.output_dim], dtype=tf.float32,
                                   initializer=self.weight_initializer)
                output_2 = tf.reshape(tf.matmul(cat_input, w), shape=(self.batch_size, self.num_station, self.output_dim))
            y_.append(output_2)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(y-y_)
        return y_, loss


    def attention_layer(self, hidden_states, adj_mx, activation=tf.nn.elu, share_weight=False):
        batch_size = hidden_states.get_shape()[0].value
        hidden_states = tf.reshape(hidden_states, (batch_size, self.num_station, -1))
        num_hidden = hidden_states.get_shape()[-1].value
        #
        adj_mx = tf.where(adj_mx>tf.zeros_like(adj_mx),
                          tf.ones_like(adj_mx),
                          tf.zeros_like(adj_mx))
        adj_mx = tf.reshape(adj_mx, (batch_size, self.num_station, -1))

        w1 = tf.get_variable('w1', shape=[num_hidden, 1], dtype=tf.float32,
                             initializer=self.weight_initializer)
        z1 = tf.get_variable('z1', shape=[num_hidden, 1], dtype=tf.float32,
                             initializer=self.weight_initializer)
        #b1 = tf.get_variable('b1', [1], dtype=tf.float32, initializer=self.const_initializer)
        #
        if share_weight:
            #w2, z2, b2 = w1, z1, b1
            w2, z2 = w1, z1
        else:
            w2 = tf.get_variable('w2', shape=[num_hidden, 1], dtype=tf.float32,
                                 initializer=self.weight_initializer)
            z2 = tf.get_variable('z2', shape=[num_hidden, 1], dtype=tf.float32,
                                 initializer=self.weight_initializer)
            #b2 = tf.get_variable('b2', [1], dtype=tf.float32, initializer=self.const_initializer)
        #
        A1 = tf.reshape(tf.matmul(tf.reshape(hidden_states, (-1, num_hidden)), w1),
                       (batch_size, self.num_station, -1))
        #A1 = tf.manip.tile(A1, (1, 1, self.num_station))
        B1 = tf.reshape(tf.matmul(tf.reshape(hidden_states, (-1, num_hidden)), z1),
                       (batch_size, 1, self.num_station))
        #B1 = tf.multiply(adj_mx, B1)
        #s_1 = activation(tf.nn.bias_add(A1+B1, b1))
        s_1 = activation(A1+B1)
        mask_1 = tf.where(adj_mx>tf.zeros_like(adj_mx),
                        tf.zeros_like(adj_mx),
                        tf.ones_like(adj_mx)*self.neg_inf)
        s_1 = tf.expand_dims(tf.nn.softmax(s_1 + mask_1, axis=1), -1)
        #
        A2 = tf.reshape(tf.matmul(tf.reshape(hidden_states, (-1, num_hidden)), w2),
                        (batch_size, self.num_station, 1))
        A2 = tf.manip.tile(A2, (1, 1, self.num_station))
        B2 = tf.reshape(tf.matmul(tf.reshape(hidden_states, (-1, num_hidden)), z2),
                        (batch_size, 1, self.num_station))
        B2 = tf.multiply(tf.transpose(adj_mx, (0, 2, 1)), B2)
        #s_2 = activation(tf.nn.bias_add(A2 + B2, b2))
        s_2 = activation(A2+B2)
        mask_2 = tf.transpose(mask_1, (0, 2, 1))
        s_2 = tf.expand_dims(tf.nn.softmax(s_2 + mask_2, axis=1), -1)
        #
        # s_mod: [batch_size, num_station, num_station]
        # hidden_states: [batch_size, num_station, num_hidden]
        # s_mod -> [batch_size, num_station, num_station, 1]
        # hidden_states -> [batch_size, num_station, num_station, num_hidden]
        tile_hidden_states = tf.manip.tile(tf.expand_dims(hidden_states, 2), (1, 1, self.num_station, 1))
        att_1 = tf.reduce_sum(tf.multiply(s_1, tile_hidden_states), -2)
        att_2 = tf.reduce_sum(tf.multiply(s_2, tile_hidden_states), -2)
        return att_1, att_2


    def att_head(self, hidden_states, adj_mx, num_units=32, activation=tf.nn.elu, share_weights=False):
        batch_size = hidden_states.get_shape()[0].value
        hidden_states = tf.reshape(hidden_states, (batch_size, self.num_station, -1))
        hidden_states = tf.layers.conv1d(hidden_states, num_units, 1, use_bias=False)
        #num_hidden = hidden_states.get_shape()[-1].value
        #
        adj_mx = tf.where(adj_mx>tf.zeros_like(adj_mx),
                          tf.ones_like(adj_mx),
                          tf.zeros_like(adj_mx))
        adj_mx = tf.reshape(adj_mx, (batch_size, self.num_station, -1))
        mask_1 = tf.where(adj_mx > tf.zeros_like(adj_mx),
                          tf.zeros_like(adj_mx),
                          tf.ones_like(adj_mx) * self.neg_inf)
        mask_2 = tf.transpose(mask_1, (0, 2, 1))
        #
        A1 = tf.layers.conv1d(hidden_states, 1, 1)
        B1 = tf.layers.conv1d(hidden_states, 1, 1)
        logits_1 = A1 + tf.transpose(B1, [0, 2, 1])
        coefs_1 = tf.nn.softmax(tf.nn.leaky_relu(logits_1) + mask_1)
        att_1 = tf.matmul(coefs_1, hidden_states)
        #
        if share_weights:
            A2, B2 = A1, B1
        else:
            A2 = tf.layers.conv1d(hidden_states, 1, 1)
            B2 = tf.layers.conv1d(hidden_states, 1, 1)
        logits_2 = A2 + tf.transpose(B2, [0, 2, 1])
        coefs_2 = tf.nn.softmax(tf.nn.leaky_relu(logits_2) + mask_2)
        att_2 = tf.matmul(coefs_2, hidden_states)
        #
        return activation(att_1), activation(att_2)

    def build_easy_model(self):
        x = tf.unstack(tf.reshape(self.x, (self.batch_size, self.input_steps, self.num_station*2)), axis=1)
        #print(len(x))
        f_all = tf.unstack(tf.reshape(self.f, (self.batch_size, self.input_steps, self.num_station*self.num_station)), axis=1)
        #print(len(f_all))
        inputs = list(zip(*(x, f_all)))
        self.cells = tf.contrib.rnn.MultiRNNCell([self.cell, self.cell_with_projection], state_is_tuple=True)
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.num_station, -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(self.y - outputs)
        return outputs, loss
    
    
    
    
    
    
    
    
