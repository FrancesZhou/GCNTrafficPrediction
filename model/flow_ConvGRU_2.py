import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell
from model.ConvGRU import Dy_Conv2DGRUCell


class flow_ConvGRU_2():
    def __init__(self, input_shape=[20,10,2], input_steps=6,
                 num_layers=2, num_units=64, kernel_shape=[3,3],
                 f_adj_mx=None,
                 batch_size=32):
        self.input_shape = input_shape
        self.input_steps = input_steps
        self.num_layers = num_layers
        self.num_units = num_units
        self.kernel_shape = kernel_shape
        self.f_adj_mx = f_adj_mx

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        # self.cells = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        # self.g_cells = tf.contrib.rnn.MultiRNNCell(g_cells, state_is_tuple=True)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0]*self.input_shape[1], self.input_shape[0]*self.input_shape[1]])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.input_shape[0], self.input_shape[1], self.input_shape[2]])


    def build_easy_model(self):
        x = tf.reshape(self.x, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        with tf.variable_scope('conv_1'):
            conv_output = tf.layers.conv2d(x, filters=self.num_units, kernel_size=self.kernel_shape, padding='SAME', activation=tf.nn.relu)
        #
        f = tf.reshape(self.f, (-1, self.input_shape[0]*self.input_shape[1], self.input_shape[0]*self.input_shape[1]))
        g_x = tf.reshape(self.x, (-1, self.input_shape[0]*self.input_shape[1], self.input_shape[2]))
        with tf.variable_scope('gconv_1'):
            gconv_output = self._gconv(num_nodes=self.input_shape[0]*self.input_shape[1],
                                       inputs=g_x, dy_adj_mx=f, output_size=self.num_units,
                                       max_diffusion_step=2, filter_type='dual_random_walk')
        coupled_conv_output = tf.add(conv_output, tf.reshape(gconv_output, (-1, self.input_shape[0], self.input_shape[1], self.num_units)))
        # second layer
        with tf.variable_scope('conv_2'):
            conv_output = tf.layers.conv2d(coupled_conv_output, filters=self.num_units, kernel_size=self.kernel_shape,
                                           padding='SAME', activation=tf.nn.relu)
        with tf.variable_scope('gconv_2'):
            gconv_output = self._gconv(num_nodes=self.input_shape[0]*self.input_shape[1],
                                   inputs=coupled_conv_output, dy_adj_mx=f, output_size=self.num_units,
                                   max_diffusion_step=2, filter_type='dual_random_walk')
        coupled_conv_output = tf.add(conv_output, tf.reshape(gconv_output, (-1, self.input_shape[0], self.input_shape[1], self.num_units)))
        coupled_conv_output = tf.transpose(tf.reshape(coupled_conv_output, (-1, self.input_steps, self.input_shape[0], self.input_shape[1], self.num_units)),
                                           (0, 2, 3, 1, 4))
        coupled_conv_output = tf.reshape(coupled_conv_output, (-1, self.input_steps, self.num_units))
        rnn_input = tf.unstack(coupled_conv_output, axis=1)
        #
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, name='lstm')
        #
        outputs, _ = tf.contrib.rnn.static_rnn(cell, rnn_input, dtype=tf.float32)
        outputs = tf.stack(outputs)
        #
        outputs = tf.reshape(outputs, (-1, self.num_units))
        final_outputs = tf.layers.dense(outputs, units=self.input_shape[-1], activation=None, kernel_initializer=self.weight_initializer)
        #
        final_outputs = tf.reshape(final_outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        final_outputs = tf.transpose(final_outputs, [1, 0, 2, 3, 4])
        loss = 2 * tf.nn.l2_loss(self.y - final_outputs)
        return final_outputs, loss

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def calculate_random_walk_matrix(self, adj_mx):
        # adj_mx: [batch_size, num_nodes, num_nodes]
        # d = tf.sparse_tensor_to_dense(tf.sparse_reduce_sum(adj_mx, 1))
        d = tf.reduce_sum(adj_mx, -1)
        d_inv = tf.where(tf.greater(d, tf.zeros_like(d)), tf.reciprocal(d), tf.zeros_like(d))
        d_mat_inv = tf.matrix_diag(d_inv)
        random_walk_mx = tf.matmul(d_mat_inv, adj_mx)
        return tf.cast(random_walk_mx, dtype=tf.float32)

    def get_supports(self, adj_mx, filter_type='dual_random_walk'):
        supports = []
        if filter_type == "random_walk":
            supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx), (0, 2, 1)))
        elif filter_type == "dual_random_walk":
            supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx), (0, 2, 1)))
            supports.append(tf.transpose(self.calculate_random_walk_matrix(tf.transpose(adj_mx, (0, 2, 1))), (0, 2, 1)))
        else:
            return None
        return supports

    def _gconv(self, num_nodes, inputs, dy_adj_mx, output_size, max_diffusion_step=2, filter_type='dual_random_walk', bias_start=0.0):
        """Graph convolution between input and the graph matrix.
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, num_nodes, -1))
        dy_adj_mx = tf.reshape(dy_adj_mx, (batch_size, num_nodes, -1))
        #
        input_size = inputs.get_shape()[2].value
        dtype = inputs.dtype
        x = inputs
        #
        len_supports = 1
        if filter_type == "dual_random_walk":
            len_supports = 2
        #
        scope = tf.get_variable_scope()
        # dy_supports = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if max_diffusion_step == 0:
                pass
            else:
                # get dynamic adj_mx
                dy_supports = self.get_supports(dy_adj_mx, filter_type=filter_type)
                # print(dy_supports)
                x0 = x
                x = tf.expand_dims(x0, axis=0)
                #
                for support in dy_supports:
                    # x0: [batch_size, num_nodes, total_arg_size]
                    # support: [batch_size, num_nodes, num_nodes]
                    # x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    # print(support.dtype)
                    # print(x0.dtype)
                    x1 = tf.matmul(support, x0)
                    x = self._concat(x, x1)
                    for k in range(2, max_diffusion_step + 1):
                        # x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x2 = 2 * tf.matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1
            # num_matrices = len(dy_supports) * self._max_diffusion_step + 1  # Adds for x itself.
            num_matrices = len_supports * max_diffusion_step + 1  # Adds for x itself.
            #
            x = tf.reshape(x, shape=[num_matrices, batch_size, num_nodes, input_size])
            x = tf.transpose(x, perm=[1, 2, 3, 0])  # (batch_size, num_nodes, input_size, order)
            #
            x = tf.reshape(x, shape=[batch_size * num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, num_nodes * output_size])





