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
                 batch_size=32):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.embedding_dim = embedding_dim
        self.ext_dim = ext_dim
        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        with tf.variable_scope('embedding'):
            if embeddings is not None:
                self.embeddings = embeddings
            else:
                self.embeddings = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.num_station, self.embedding_dim]), name='embeddings')
        with tf.variable_scope('lstm'):
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.embedding_dim)
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
        hidden_state = tf.zeros([self.num_station, self.embedding_dim])
        current_state = tf.zeros([self.num_station, self.embedding_dim])
        state = hidden_state, current_state
        y_ = []
        for i in xrange(self.input_steps):
            # for each step
            # ------------------- transition-in gate & transition-out gate -------------
            f = f_all[i]
            f_in_sum = tf.tile(tf.expand_dims(tf.reduce_sum(f, 0), 0), [self.num_station, 1])
            f_in_gate = tf.where(f_in_sum>0, tf.divide(f, f_in_sum), tf.ones_like(f))
            f_out_sum = tf.tile(tf.expand_dims(tf.reduce_sum(f, 1), -1), [1, self.num_station])
            f_out_gate = tf.transpose(tf.where(f_out_sum>0, tf.divide(f, f_out_sum), tf.ones_like(f)))
            #f_in_gate = tf.divide(f, tf.reduce_sum(f, 0, keepdims=True))
            #f_out_gate = tf.transpose(tf.divide(f, tf.reduce_sum(f, 1, keepdims=True)))
            f_in = tf.multiply(tf.tile(tf.expand_dims(x[i, :, 0], axis=0), [self.num_station, 1]),
                               f_out_gate)
            f_out = tf.multiply(tf.tile(tf.expand_dims(x[i, :, 1], axis=0), [self.num_station, 1]),
                                f_in_gate)
            # f_in, f_out: [num_station, num_station]
            current_step_batch = tf.concat((f_in, f_out), axis=-1)
            #current_step_batch = f
            # current_step_batch: [num_station, 2*num_station]
            output, state = self.lstm(current_step_batch, state)
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

'''
    def predict(self):
        x = tf.transpose(self.x, [1, 0, 2, 3])
        y_test = self.y_test
        f = tf.transpose(self.f_test, [1, 0, 2, 3])
        # f_input = f[:self.input_steps, :, :, :]
        # f_output = f[self.input_steps:, :, :, :]
        # x: [input_steps, batch_size, num_station, 2]
        # y: [output_steps, batch_size, num_station, 2]
        # f: [input_steps+output_steps, batch_size, num_station, num_station]

        tile_embeddings = tf.tile(tf.expand_dims(self.embeddings, axis=0),
                                  [self.batch_size * self.num_station, 1, 1])
        tile_embeddings = tf.reshape(tile_embeddings,
                                     [self.batch_size, self.num_station, self.num_station, self.embedding_dim])

        #self.lstm = tf.contrib.rnn.BasicLSTMCell(self.embedding_dim)
        # Initial state of the LSTM memory.
        # hidden_state = tf.zeros([self.batch_size, self.lstm.state_size])
        # current_state = tf.zeros([self.batch_size, self.lstm.state_size])
        hidden_state = tf.zeros([self.batch_size, self.embedding_dim])
        current_state = tf.zeros([self.batch_size, self.embedding_dim])
        state = hidden_state, current_state
        y_ = []
        next_input = []
        for i in xrange(self.input_steps+self.output_steps-1):
            # for each step
            if i > self.input_steps-1:
                current_step_batch = next_input
            else:
                current_step_batch = x[i]
            output, state = self.lstm(tf.reshape(current_step_batch, [self.batch_size, -1]), state)
            # output: [batch_size, state_size]
            if i > self.input_steps-2:
                # ------------------- dynamic spatial dependency ------------------------
                if self.dynamic_spatial:
                    f_embedding = tf.multiply(tf.expand_dims(f[i-(self.input_steps-1)], axis=-1),
                                              tile_embeddings)
                    # f_embedding: [batch_size, num_station, num_station, embedding_dim]
                    alpha = tf.reshape(tf.reduce_sum(tf.multiply(tf.expand_dims(output, axis=1),
                                                                 tf.reshape(f_embedding,
                                                                            [self.batch_size, -1, self.embedding_dim])),
                                                     axis=-1, keep_dims=True),
                                       [self.batch_size, self.num_station, self.num_station, 1])
                    # alpha: [batch_size, num_station, num_station, 1]
                    alpha = tf.nn.softmax(alpha, axis=-2)
                    Dy_s = tf.reduce_sum(tf.multiply(f_embedding, alpha), axis=-2)
                    # Dy_s: [batch_size, num_station, embedding_dim]
                else:
                    Dy_s = tf.constant(0.0, dtype=tf.float32, shape=[self.batch_size, self.num_station, self.embedding_dim])
                # ------------------- output ---------------------
                # hidden_y = relu(w1*Dy_s + w2*output + b)
                hidden_out_dim = self.hidden_dim
                # hidden_out_dim = 2
                hidden_y = self.fusion((tf.tile(tf.expand_dims(output, axis=1), [1, self.num_station, 1]),),
                                       out_dim=hidden_out_dim, reuse=tf.AUTO_REUSE)
                next_input = tf.nn.relu(tf.add(tf.matmul(self.out_hidden_w, tf.reshape(hidden_y, [-1, self.hidden_dim])),
                           self.out_hidden_b))
                #next_input = hidden_y
                next_input = tf.reshape(next_input, [self.batch_size, self.num_station, -1])
                y_.append(next_input)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2 * tf.nn.l2_loss(y_test - y_)
        return y_, loss
'''

