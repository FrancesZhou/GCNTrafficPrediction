import sys
import numpy as np
import cPickle as pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *


class DyST():
    def __init__(self, num_station, input_steps, output_steps,
                 embedding_dim=100, embeddings=None,
                 hidden_dim=64,
                 batch_size=32,
                 dynamic_spatial=0):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.batch_size = batch_size
        self.dynamic_spatial = dynamic_spatial

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()
        if embeddings is not None:
            self.embeddings = embeddings
        else:
            self.embeddings = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.num_station, self.embedding_dim]), name='embeddings')

        self.x = tf.placeholder(tf.float32, [batch_size, self.input_steps, self.num_station, 2])
        self.y_train = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[batch_size, self.input_steps, self.num_station, 2]), shape=[batch_size, self.input_steps, self.num_station, 2])
        self.f_train = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[batch_size, self.input_steps, self.num_station, self.num_station]), shape=[batch_size, self.input_steps, self.num_station, self.num_station])
        #
        self.y_test = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[batch_size, self.output_steps, self.num_station, 2]), shape=[batch_size, self.output_steps, self.num_station, 2])
        self.f_test = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32, shape=[batch_size, self.output_steps, self.num_station, self.num_station]), shape=[batch_size, self.output_steps, self.num_station, self.num_station])

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

    def build_model(self):
        x = tf.transpose(self.x, [1, 0, 2, 3])
        y_train = self.y_train
        f = tf.transpose(self.f_train, [1, 0, 2, 3])
        # x: [input_steps, batch_size, num_station, 2]
        # y: [output_steps, batch_size, num_station, 2]
        # f: [input_steps+output_steps, batch_size, num_station, num_station]

        tile_embeddings = tf.tile(tf.expand_dims(self.embeddings, axis=0),
                                  [self.batch_size*self.num_station, 1, 1])
        tile_embeddings = tf.reshape(tile_embeddings,
                                     [self.batch_size, self.num_station, self.num_station, self.embedding_dim])

        self.lstm = tf.contrib.rnn.BasicLSTMCell(self.embedding_dim)
        # Initial state of the LSTM memory.
        #hidden_state = tf.zeros([self.batch_size, self.lstm.state_size])
        #current_state = tf.zeros([self.batch_size, self.lstm.state_size])
        hidden_state = tf.zeros([self.batch_size, self.embedding_dim])
        current_state = tf.zeros([self.batch_size, self.embedding_dim])
        state = hidden_state, current_state
        y_ = []
        for i in xrange(self.input_steps):
            # for each step
            current_step_batch = x[i]
            output, state = self.lstm(tf.reshape(current_step_batch, [self.batch_size, -1]), state)
            # output: [batch_size, state_size]
            # ------------------- dynamic spatial dependency ------------------------
            if self.dynamic_spatial:
                f_embedding = tf.multiply(tf.expand_dims(f[i], axis=-1),
                                          tile_embeddings)
                # f_embedding: [batch_size, num_station, num_station, embedding_dim]
                alpha = tf.reshape(tf.reduce_sum(tf.multiply(tf.expand_dims(output, axis=1),
                                                             tf.reshape(f_embedding, [self.batch_size, -1, self.embedding_dim])),
                                                 axis=-1, keep_dims=True), [self.batch_size, self.num_station, self.num_station, 1])
                # alpha: [batch_size, num_station, num_station, 1]
                alpha = tf.nn.softmax(alpha, axis=-2)
                Dy_s = tf.reduce_sum(tf.multiply(f_embedding, alpha), axis=-2)
                # Dy_s: [batch_size, num_station, embedding_dim]
            else:
                Dy_s = tf.constant(0.0, dtype=tf.float32, shape=[self.batch_size, self.num_station, self.embedding_dim])
            # ------------------- output ---------------------
            # hidden_y = relu(w1*Dy_s + w2*output + b)
            hidden_y = self.fusion((Dy_s, tf.tile(tf.expand_dims(output, axis=1), [1, self.num_station, 1])),
                                      out_dim=self.hidden_dim, reuse=tf.AUTO_REUSE)
            # next_output = relu(w*hidden_y + b)
            next_output = tf.layers.dense(tf.reshape(hidden_y, [-1, self.hidden_dim]), 2, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
            next_output = tf.reshape(next_output, [self.batch_size, self.num_station, -1])
            y_.append(next_output)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(y_train-y_)
        return loss

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
                hidden_y = self.fusion((Dy_s, tf.tile(tf.expand_dims(output, axis=1), [1, self.num_station, 1])),
                                       out_dim=self.hidden_dim, reuse=tf.AUTO_REUSE)
                next_input = tf.layers.dense(tf.reshape(hidden_y, [-1, self.hidden_dim]), 2, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
                next_input = tf.reshape(next_input, [self.batch_size, self.num_station, -1])
                y_.append(next_input)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2 * tf.nn.l2_loss(y_test - y_)
        return y_, loss


