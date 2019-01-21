from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops

import utils


class Coupled_DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, num_proj=None,
                 input_dim=None, dy_adj=1, dy_filter=0, output_dy_adj=False,
                 activation=tf.nn.tanh, reuse=None, filter_type="dual_random_walk", use_gc_for_ru=True):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        #
        :param input_dim: num_nodes*input_channels if dy_adj=1
        :param dy_adj: whether to use dynamic adjacent matrix (input_dim should be given if dy_adj=1)
        :param dy_filter: whether to use dynamic generated filter
        #
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(Coupled_DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self.filter_type = filter_type
        # self.dy_adj = dy_adj
        # self.dy_filter = dy_filter
        self.output_dy_adj = output_dy_adj
        self._num_nodes = num_nodes
        self._input_dim = input_dim
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._supports = []
        #

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        if self.filter_type == "dual_random_walk":
            self._len_supports = 2
        else:
            self._len_supports = 1

        if adj_mx is not None:
            # for fixed adjacent matrix
            if self.filter_type == "random_walk":
                self._supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx)))
            elif self.filter_type == "dual_random_walk":
                self._supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx)))
                self._supports.append(tf.transpose(self.calculate_random_walk_matrix(tf.transpose(adj_mx))))


    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    #def __call__(self, inputs, adj_mx, state, scope=None):
    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param
        - #inputs: (input, adj_mx)
        inputs: (B, num_nodes * input_dim)
        adj_mx: (B, num_nodes * num_nodes)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        if self._input_dim is not None:
            whole_input_dim = inputs.get_shape().as_list()
            dy_adj_dim = whole_input_dim[-1] - self._input_dim * self._num_nodes
            if dy_adj_dim>0:
                _input, dy_adj_mx = tf.split(inputs, num_or_size_splits=[self._input_dim*self._num_nodes, dy_adj_dim], axis=-1)
                inputs = _input
            else:
                #print('There is no input dynamic flow to generate dynamic adjacent matrix.')
                dy_adj_mx = None
        else:
            dy_adj_mx = None
        #
        # ------- dcgru ------
        with tf.variable_scope('dcgru', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('gates', reuse=tf.AUTO_REUSE):
                value = self._gconv(inputs=inputs, state=state, dy_adj_mx=None,
                                    output_size=2*self._num_units)
                value = tf.reshape(value, (-1, self._num_nodes, 2*self._num_units))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))
        # ------- flow-gcn --------
        with tf.variable_scope('flow-gcn', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('gates', reuse=tf.AUTO_REUSE):
                f_value = self._gconv(inputs=inputs, state=state, dy_adj_mx=dy_adj_mx,
                                      output_size=2*self._num_units, bias=False)
                f_value = tf.reshape(f_value, (-1, self._num_nodes, 2*self._num_units))
                f_r, f_u = tf.split(value=f_value, num_or_size_splits=2, axis=-1)
                f_r = tf.reshape(f_r, (-1, self._num_nodes * self._num_units))
                f_u = tf.reshape(f_u, (-1, self._num_nodes * self._num_units))
        #
        couple_r = tf.nn.sigmoid(r + f_r)
        couple_u = tf.nn.sigmoid(u + f_u)
        with tf.variable_scope('candidate', reuse=tf.AUTO_REUSE):
            c = self._gconv(inputs=inputs, state=couple_r * state, dy_adj_mx=None,
                            output_size=self._num_units)
            f_c = self._gconv(inputs=inputs, state=couple_r * state, dy_adj_mx=dy_adj_mx,
                              output_size=self._num_units)
        if self._activation is not None:
            couple_c = self._activation(c + f_c)
        else:
            couple_c = c + f_c
        # -------- coupled dcgru and flow-gcn -------
        output = new_state = couple_u * state + (1 - couple_u) * couple_c
        if self._num_proj is not None:
            with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
                w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                batch_size = inputs.get_shape()[0].value
                output = tf.reshape(new_state, shape=(-1, self._num_units))
                output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        if self.output_dy_adj:
            #print(output)
            #print(dy_adj_mx)
            output = tf.concat([output, dy_adj_mx], axis=-1)
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, dy_adj_mx, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    def calculate_random_walk_matrix(self, adj_mx):
        # adj_mx: [batch_size, num_nodes, num_nodes]
        # d = tf.sparse_tensor_to_dense(tf.sparse_reduce_sum(adj_mx, 1))
        d = tf.reduce_sum(adj_mx, -1)
        d_inv = tf.where(tf.greater(d, tf.zeros_like(d)) , tf.reciprocal(d), tf.zeros_like(d))
        d_mat_inv = tf.matrix_diag(d_inv)
        random_walk_mx = tf.matmul(d_mat_inv, adj_mx)
        return tf.cast(random_walk_mx, dtype=tf.float32)

    def calculate_random_walk_matrix_2d(self, adj_mx):
        # adj_mx: [num_nodes, num_nodes]
        adj_mx = sp.coo_matrix(adj_mx)
        d = np.array(adj_mx.sum(1))
        d_inv = np.power(d, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
        return random_walk_mx


    def get_supports(self, adj_mx):
        supports = []
        if self.filter_type == "random_walk":
            supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx), (0, 2, 1)))
        elif self.filter_type == "dual_random_walk":
            supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx), (0, 2, 1)))
            supports.append(tf.transpose(self.calculate_random_walk_matrix(tf.transpose(adj_mx, (0, 2, 1))), (0, 2, 1)))
        else:
            return None
        '''
        dy_supports = []
        for support in supports:
            dy_supports.append(self._build_sparse_matrix(support))
        return dy_supports
        '''
        return supports

    def _gconv(self, inputs, state, dy_adj_mx, output_size, bias=True, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))

        if dy_adj_mx is not None:
            #print('dy_adj_mx is right.')
            dy_adj_mx = tf.reshape(dy_adj_mx, (batch_size, self._num_nodes, -1))
        else:
            #print('No dynamic flow input to generate dynamic adjacent matrix.')
            dy_adj_mx = None
        
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        #
        scope = tf.get_variable_scope()
        #dy_supports = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self._max_diffusion_step == 0:
                pass
            else:
                # get dynamic adj_mx
                if dy_adj_mx is None:
                    dy_supports = self._supports
                    #print(dy_supports)
                    x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
                    x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
                    x = tf.expand_dims(x0, axis=0)
                else:
                    #print(dy_adj_mx)
                    dy_supports = self.get_supports(dy_adj_mx)
                    #print(dy_supports)
                    x0 = x
                    x = tf.expand_dims(x0, axis=0)
                #
                for support in dy_supports:
                    # x0: [batch_size, num_nodes, total_arg_size]
                    # support: [batch_size, num_nodes, num_nodes]
                    #x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    #print(support.dtype)
                    #print(x0.dtype)
                    x1 = tf.matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        #x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x2 = 2 * tf.matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            #num_matrices = len(dy_supports) * self._max_diffusion_step + 1  # Adds for x itself.
            num_matrices = self._len_supports * self._max_diffusion_step + 1  # Adds for x itself.
            #
            if dy_adj_mx is None:
                x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
                x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            else:
                x = tf.reshape(x, shape=[num_matrices, batch_size, self._num_nodes, input_size])
                x = tf.transpose(x, perm=[1, 2, 3, 0])  # (batch_size, num_nodes, input_size, order)
            #
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
            if bias:
                biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                         initializer=tf.constant_initializer(bias_start, dtype=dtype))
                x = tf.nn.bias_add(x, biases)

        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])
