import sys
import numpy as np
import cPickle as pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *


class DyST():
    def __init__(self, num_station, input_steps, output_steps,
                 embedding_dim=100, embeddings=None,
                 ext_dim=7,
                 hidden_dim=64,
                 batch_size=32,
                 dynamic_spatial=0):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.embedding_dim = embedding_dim
        self.ext_dim = ext_dim
        self.hidden_dim = hidden_dim

        self.batch_size = batch_size
        self.dynamic_spatial = dynamic_spatial

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        with tf.variable_scope('embedding', reuse=tf.AUTO_REUSE):
            if embeddings is not None:
                self.embeddings = embeddings
            else:
                self.embeddings = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.num_station, self.embedding_dim]), name='embeddings')
        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            self.lstm = tf.contrib.rnn.BasicLSTMCell(self.embedding_dim)
        with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
            self.w_1 = tf.get_variable(shape=[self.num_station, self.num_station], initializer=self.weight_initializer, name='w_1')
            self.w_2 = tf.get_variable(shape=[self.num_station, self.num_station], initializer=self.weight_initializer, name='w_2')
            self.w_3 = tf.get_variable(shape=[self.num_station, self.num_station], initializer=self.weight_initializer,
                                       name='w_3')
            self.w_4 = tf.get_variable(shape=[self.num_station, self.num_station], initializer=self.weight_initializer,
                                       name='w_4')
            self.w_h_in = tf.get_variable(shape=[self.num_station, self.embedding_dim], initializer=self.weight_initializer, name='w_h_in')
            self.w_h_out = tf.get_variable(shape=[self.num_station, self.embedding_dim], initializer=self.weight_initializer, name='w_h_out')
            self.w_e_in = tf.get_variable(shape=[self.ext_dim, self.num_station], initializer=self.weight_initializer, name='w_e_in')
            self.w_e_out = tf.get_variable(shape=[self.ext_dim, self.num_station], initializer=self.weight_initializer, name='w_e_out')
            self.b = tf.get_variable(shape=[self.num_station,], initializer=self.const_initializer, name='b')

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.e = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.ext_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])

    def fusion(self, data, out_dim, reuse=True):
        out_shape = data[0].get_shape().as_list()
        out_shape[-1] = out_dim
        shape = [np.prod(out_shape[:-1]), out_dim]
        out = tf.constant(0.0, dtype=tf.float32, shape=shape, name='fusion_output')
        for i in xrange(len(data)):
            with tf.variable_scope('fusion_{0}'.format(i), reuse=reuse):
                dim = data[i].get_shape().as_list()[-1]
                d = tf.reshape(data[i], [-1, dim])
                w = tf.get_variable('w', [dim, out_dim])
                out = tf.add(out, tf.matmul(d, w))
        with tf.variable_scope('bias', reuse=reuse):
            b = tf.get_variable('b', [out_dim])
            out = tf.add(out, b)
        out = tf.nn.relu(out)
        return tf.reshape(out, out_shape)

    def attention(self, f_one_zero, corr, embeddings):
        alpha = tf.multiply(tf.expand_dims(corr, axis=1), f_one_zero)  # [batch_size, num_station, num_station]
        alpha = tf.reshape(alpha, (-1, self.num_station))
        #alpha = tf.nn.softmax(alpha, axis=-1)
        alpha = tf.contrib.sparsemax.sparsemax(alpha)
        # alpha: [batch_size*num_station, num_station]
        # embeddings: [num_station, embedding_dim]
        context = tf.reduce_sum(tf.multiply(tf.expand_dims(alpha, -1), embeddings), axis=-2)
        context = tf.reshape(context, (-1, self.num_station, self.embedding_dim))
        return context

    def build_model(self):
        x = tf.transpose(self.x, [1, 0, 2, 3])
        f_all = tf.transpose(self.f, [1, 0, 2, 3])
        e_all = tf.transpose(self.e, [1, 0, 2])
        y = self.y
        # x: [input_steps, batch_size, num_station, 2]
        # y: [input_steps, batch_size, num_station, 2]
        # f: [input_steps, batch_size, num_station, num_station]
        # Initial state of the LSTM memory.
        #hidden_state = tf.zeros([self.batch_size, self.lstm.state_size])
        #current_state = tf.zeros([self.batch_size, self.lstm.state_size])
        hidden_state = tf.zeros([self.batch_size, self.embedding_dim])
        current_state = tf.zeros([self.batch_size, self.embedding_dim])
        state = hidden_state, current_state
        y_ = []
        for i in xrange(self.input_steps):
            # for each step
            f = f_all[i]
            current_step_batch = x[i]
            output, state = self.lstm(tf.reshape(current_step_batch, [self.batch_size, -1]), state)
            # output: [batch_size, state_size]
            # ------------------ dynamic context ----------------
            # compute alpha
            # embeddings: [num_station, embedding_dim]
            corr = tf.matmul(output, tf.transpose(self.embeddings))  # [batch_size, num_station]
            f_in_one_zero = tf.cast(tf.greater(f, tf.zeros_like(f)), tf.float32)  # [batch_size, num_station, num_station]
            f_out_one_zero = tf.cast(tf.greater(tf.transpose(f, (0, 2, 1)), tf.zeros_like(tf.transpose(f, (0,2,1)))), tf.float32)
            cxt_in = self.attention(f_in_one_zero, corr, self.embeddings)
            cxt_out = self.attention(f_out_one_zero, corr, self.embeddings)
            # cxt_in: [batch_size, num_station, embedding_dim]
            #cxt = tf.concat((cxt_in, cxt_out), axis=-1)
            # ------------------ dynamic spatial dependency -----------------
            # f: [batch_size, num_station, num_station]
            f_in_sum = tf.tile(tf.reduce_sum(f, 1, keepdims=True), [1, self.num_station, 1])
            #f_in_gate = tf.where(f_in_sum > 0, tf.divide(f, f_in_sum), f)
            f_in_gate = tf.where(f_in_sum > 0, tf.ones_like(f), f)
            f_out_sum = tf.tile(tf.reduce_sum(f, 2, keepdims=True), [1, 1, self.num_station])
            #f_out_gate = tf.transpose(tf.where(f_out_sum > 0, tf.divide(f, f_out_sum), f), (0, 2, 1))
            f_out_gate = tf.transpose(tf.where(f_out_sum > 0, tf.ones_like(f), f), (0, 2, 1))
            # check-out
            x_in = x[i, :, :, 0]
            x_out = x[i, :, :, 1]
            # [batch_size, num_station]
            out_1 = tf.squeeze(tf.matmul(tf.multiply(f_in_gate, self.w_1), tf.expand_dims(x_in, -1))) + tf.matmul(x_out, self.w_2)
            out_2 = tf.reduce_sum(tf.multiply(cxt_out, self.w_h_out), axis=-1)
            #
            e = e_all[i]
            out_3 = tf.matmul(e, self.w_e_out)
            next_out = out_1 + out_2 + out_3
            # check-in
            in_1 = tf.squeeze(tf.matmul(tf.multiply(f_out_gate, self.w_3), tf.expand_dims(x_out, -1))) + tf.matmul(x_in, self.w_4)
            in_2 = tf.reduce_sum(tf.multiply(cxt_in, self.w_h_in), axis=-1)
            #
            in_3 = tf.matmul(e, self.w_e_in)
            next_in = in_1 + in_2 + in_3
            next_output = tf.concat((tf.expand_dims(next_in, -1), tf.expand_dims(next_out, -1)), -1)
            y_.append(next_output)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(y-y_)
        return y_, loss
