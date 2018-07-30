import sys
import numpy as np
import cPickle as pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *


class DyST2():
    def __init__(self, num_station, input_steps, output_steps,
                 embedding_dim=100, embeddings=None,
                 ext_dim=7,
                 hidden_dim=64,
                 batch_size=32,
                 topk=10):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.embedding_dim = embedding_dim
        self.ext_dim = ext_dim
        self.hidden_dim = hidden_dim

        self.batch_size = batch_size
        self.topk = topk

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        with tf.variable_scope('embedding'):
            if embeddings is not None:
                self.embeddings = embeddings
            else:
                self.embeddings = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.num_station, self.embedding_dim]), name='embeddings')
        with tf.variable_scope('rnn'):
            #self.rnn = tf.contrib.rnn.BasicLSTMCell(self.embedding_dim)
            self.rnn = tf.contrib.rnn.GRUCell(self.embedding_dim)
        with tf.variable_scope('output'):
            self.out_hidden_w = tf.get_variable(shape=[self.hidden_dim, 2], initializer=self.weight_initializer, name='out_hidden_w')
            self.out_hidden_b = tf.get_variable(shape=[2,], initializer=self.const_initializer, name='out_hidden_b')

        self.x = tf.placeholder(tf.float32, [self.input_steps, self.num_station, 2])
        self.f = tf.placeholder(tf.float32, [self.input_steps, self.num_station, self.num_station])
        self.e = tf.placeholder(tf.float32, [self.input_steps, self.ext_dim])
        self.y = tf.placeholder(tf.float32, [self.input_steps, self.num_station, 2])

    def fusion(self, data, out_dim, reuse=True):
        out = tf.constant(0.0, dtype=tf.float32, shape=[self.num_station, out_dim], name='fustion_output')
        for i in xrange(len(data)):
            with tf.variable_scope('fusion_{0}'.format(i), reuse=reuse):
                dim = data[i].get_shape().as_list()[-1]
                w = tf.get_variable('w', [dim, out_dim])
                out = tf.add(out, tf.matmul(data[i], w))
        with tf.variable_scope('bias', reuse=reuse):
            b = tf.get_variable('b', [out_dim])
            out = tf.add(out, b)
        out = tf.nn.relu(out)
        #out = tf.sigmoid(out)
        return out

    def attention(self, f_one_zero, corr, tile_embeddings):
        corr = tf.multiply(corr, f_one_zero)  # [num_station, num_station]
        alpha = tf.where(corr > 0, x=tf.exp(corr), y=tf.zeros_like(corr))
        alpha_sum = tf.reduce_sum(alpha, axis=-1)
        alpha = tf.where(alpha_sum > 0, x=alpha / alpha_sum, y=tf.zeros_like(alpha))  # [num_station, num_station]
        # alpha = tf.divide(alpha, tf.reduce_sum(alpha, axis=-1)) # [num_station, num_station]
        context = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, axis=-1), tile_embeddings), axis=-2)
        return context

    def get_top_k(self, f, topk, x):
        _, indices = tf.nn.top_k(f, topk)
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk])
        full_indices = tf.stack([my_range_repeated, indices], axis=2)
        full_indices = tf.reshape(full_indices, [-1, 2])
        x_values = tf.gather_nd(x, full_indices)
        f_topk = tf.sparse_to_dense(full_indices, tf.shape(f), tf.reshape(x_values, [-1]), default_value=0., validate_indices=False)
        return f_topk

    def build_model(self):
        x = self.x
        y = self.y
        f_all = self.f
        e = self.e
        # x: [input_steps, num_station, 2]
        # f: [input_steps, num_station, num_station]
        # e: [input_steps, ext_dim]
        # y: [input_steps, num_station, 2]
        #
        # Initial state of the LSTM memory.
        #hidden_state = tf.zeros([self.batch_size, self.lstm.state_size])
        #current_state = tf.zeros([self.batch_size, self.lstm.state_size])
        '''
        hidden_state = tf.zeros([self.num_station, self.embedding_dim])
        current_state = tf.zeros([self.num_station, self.embedding_dim])
        state = hidden_state, current_state
        '''
        state = tf.zeros([self.num_station, self.embedding_dim])
        y_ = []
        for i in xrange(self.input_steps):
            # for each step
            # ------------------- transition-in gate & transition-out gate -------------
            f = f_all[i]
            #'''
            f_in_sum = tf.tile(tf.expand_dims(tf.reduce_sum(f, 0), 0), [self.num_station, 1])
            #f_in_gate = tf.where(f_in_sum > 0, tf.divide(f, f_in_sum), f)
            f_in_gate = tf.where(f_in_sum > 0, tf.ones_like(f), f)
            f_out_sum = tf.tile(tf.expand_dims(tf.reduce_sum(f, 1), -1), [1, self.num_station])
            #f_out_gate = tf.transpose(tf.where(f_out_sum > 0, tf.divide(f, f_out_sum), f))
            f_out_gate = tf.transpose(tf.where(f_out_sum > 0, tf.ones_like(f), f))
            #f_in_gate = tf.divide(f, tf.reduce_sum(f, 0, keepdims=True))
            #f_out_gate = tf.transpose(tf.divide(f, tf.reduce_sum(f, 1, keepdims=True)))
            f_in = tf.multiply(tf.tile(tf.expand_dims(x[i, :, 0], axis=0), [self.num_station, 1]),
                               f_out_gate)
            f_out = tf.multiply(tf.tile(tf.expand_dims(x[i, :, 1], axis=0), [self.num_station, 1]),
                                f_in_gate)
            # ---- top k -----
            f_in = self.get_top_k(tf.transpose(f), self.topk, f_in)
            f_out = self.get_top_k(f, self.topk, f_out)
            # ----
            # f_in, f_out: [num_station, num_station]
            current_step_batch = tf.concat((f_in, f_out), axis=-1)
            #'''
            #current_step_batch = tf.tile(tf.expand_dims(tf.reshape(x[i], [-1,]), axis=0), [self.num_station, 1])
            #current_step_batch = f
            # current_step_batch: [num_station, 2*num_station]
            output, state = self.rnn(current_step_batch, state)
            # output: [num_station, state_size]
            # ------------------- dynamic context ------------------------
            # compute alpha
            tile_embeddings = tf.tile(tf.expand_dims(self.embeddings, axis=0),
                                      [self.num_station, 1, 1])  # [num_station, num_station, embedding_dim]
            corr = tf.matmul(output, tf.transpose(self.embeddings)) # [num_station(batch_size), num_station]
            f_in_one_zero = tf.cast(tf.greater(f, tf.zeros_like(f)), tf.float32) # [num_station, num_station]
            f_out_one_zero = tf.cast(tf.greater(tf.transpose(f), tf.zeros_like(tf.transpose(f))), tf.float32)
            cxt_in = self.attention(f_in_one_zero, corr, tile_embeddings)
            cxt_out = self.attention(f_out_one_zero, corr, tile_embeddings)
            cxt = tf.concat((cxt_in, cxt_out), axis=-1)
            # cxt: [num_station, 2*embedding_dim]
            # ------------------- output ---------------------
            ext = e[i] # [ext_dim]
            # hidden_y = relu(w1*Dy_s + w2*output + b)
            #hidden_out_dim = self.hidden_dim
            hidden_out_dim = 2
            hidden_y = self.fusion((output, cxt, tf.tile(tf.expand_dims(ext, axis=0), [self.num_station, 1])),
                                      out_dim=hidden_out_dim, reuse=tf.AUTO_REUSE)
            #hidden_y: [num_station, hidden_out_dim]
            next_output = hidden_y
            #next_output = tf.nn.relu(tf.add(tf.matmul(hidden_y, self.out_hidden_w),
            #                                self.out_hidden_b))
            y_.append(next_output)
        y_ = tf.stack(y_)
        #y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(y-y_)
        return y_, loss



