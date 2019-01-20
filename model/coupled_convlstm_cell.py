# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import collections
# import math
import numpy as np
import tensorflow as tf
# from tensorflow.contrib.compiler import jit
# from tensorflow.contrib.layers.python.layers import layers
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# from tensorflow.python.framework import constant_op
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import op_def_registry
# from tensorflow.python.framework import ops
# from tensorflow.python.layers import base as base_layer
# from tensorflow.python.ops import gen_array_ops
# from tensorflow.python.ops import clip_ops
# from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
# from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
# from tensorflow.python.ops import random_ops
# from tensorflow.python.platform import tf_logging as logging
# from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

from model.dcrnn_cell import DCGRUCell

class Coupled_Conv2DLSTMCell(rnn_cell_impl.RNNCell):
    """Convolutional LSTM recurrent network cell.

    https://arxiv.org/pdf/1506.04214v1.pdf
    """

    def __init__(self,
                 input_shape,
                 output_channels,
                 kernel_shape,
                 input_dim=None, dy_adj=0, dy_filter=0, output_dy_adj=0,
                 max_diffusion_step=2,
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=1.0,
                 initializers=None,
                 name="conv_lstm_cell"):
        """Construct ConvLSTMCell.

        Args:
            conv_ndims: Convolution dimensionality (1, 2 or 3).
            input_shape: Shape of the input as int tuple, excluding the batch size.
            output_channels: int, number of output channels of the conv LSTM.
            kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
            use_bias: (bool) Use bias in convolutions.
            skip_connection: If set to `True`, concatenate the input to the
            output of the conv LSTM. Default: `False`.
            forget_bias: Forget bias.
            initializers: Unused.
            name: Name of the module.

        Raises:
            ValueError: If `skip_connection` is `True` and stride is different from 1
            or if `input_shape` is incompatible with `conv_ndims`.
        """
        super(Coupled_Conv2DLSTMCell, self).__init__(name=name)

        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape

        self._input_dim = input_dim
        # self.dy_adj = dy_adj
        # self.dy_filter = dy_filter
        self.output_dy_adj = output_dy_adj

        # for coupled flow-gcn module
        self._max_diffusion_step = max_diffusion_step
        self._num_nodes = input_shape[0]*input_shape[1]
        # self.flow_gcn = DCGRUCell(output_channels, adj_mx=None, max_diffusion_step=2, num_nodes=input_shape[0]*input_shape[1],
        #                           input_dim=input_dim, dy_adj=1, dy_filter=0, output_dy_adj=output_dy_adj)
        #

        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._skip_connection = skip_connection

        self._total_output_channels = output_channels
        if self._skip_connection:
            self._total_output_channels += self._input_shape[-1]

        state_size = tensor_shape.TensorShape(
            self._input_shape[:-1] + [self._output_channels])
        self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
        self._output_size = tensor_shape.TensorShape(
            self._input_shape[:-1] + [self._total_output_channels])
        
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, state, scope=None):
        if self._input_dim is not None:
            whole_input_dim = inputs.get_shape().as_list()
            dy_f_dim = whole_input_dim[-1] - self._input_dim
            if dy_f_dim > 0:
                _input, dy_f = tf.split(inputs, num_or_size_splits=[self._input_dim, dy_f_dim], axis=-1)
                #print('we have dynamic flow data.')
                inputs = _input
            else:
                dy_f = None
        else:
            dy_f = None
        #
        cell, hidden = state
        # ------ convlstm --------
        new_hidden = self._conv(args=[inputs, hidden], filter_size=self._kernel_shape,
                           num_features=4 * self._output_channels, bias=False, bias_start=0)
        gates = array_ops.split(value=new_hidden, num_or_size_splits=4, axis=3)
        input_gate, new_input, forget_gate, output_gate = gates
        #
        # ------ flow-gcn --------
        f_new_hidden = self._gconv(inputs=inputs, state=hidden, dy_adj_mx=dy_f,
                                            output_size=4*self._output_channels)
        f_new_hidden = tf.reshape(f_new_hidden, [-1, self.input_shape[0], self.input_shape[1], 4*self._output_channels])
        f_gates = array_ops.split(values=f_new_hidden, num_or_size_splits=4, axis=-1)
        f_input_gate, f_new_input, f_forget_gate, f_output_gate = f_gates
        # ------ couple convlstm and flow-gcn -------
        new_cell = math_ops.sigmoid(forget_gate + f_forget_gate + self._forget_bias) * cell
        new_cell += math_ops.sigmoid(input_gate + f_input_gate) * math_ops.tanh(new_input + f_new_input)
        output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate + f_output_gate)
        #
        # new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
        # new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
        # output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)
        #
        if self._skip_connection:
            output = array_ops.concat([output, inputs], axis=-1)
        new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
        if self.output_dy_adj>0:
            print(output.get_shape().as_list())
            print(dy_f.get_shape().as_list())
            output = tf.concat([output, dy_f], axis=-1)
        return output, new_state

    def _conv(self, args, filter_size, num_features, bias, bias_start=0.0):
        """Convolution.

        Args:
            args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
            batch x n, Tensors.
            filter_size: int tuple of filter height and width.
            num_features: int, number of features.
            bias: Whether to use biases in the convolution layer.
            bias_start: starting value to initialize the bias; 0 by default.

        Returns:
            A 3D, 4D, or 5D Tensor with shape [batch ... num_features]

        Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
        """

        # Calculate the total size of arguments on dimension 1.
        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        shape_length = len(shapes[0])
        for shape in shapes:
            if len(shape) not in [3, 4, 5]:
                raise ValueError("Conv Linear expects 3D, 4D "
                                 "or 5D arguments: %s" % str(shapes))
            if len(shape) != len(shapes[0]):
                raise ValueError("Conv Linear expects all args "
                                 "to be of same Dimension: %s" % str(shapes))
            else:
                total_arg_size_depth += shape[-1]
        dtype = [a.dtype for a in args][0]

        # determine correct conv operation
        if shape_length == 3:
            conv_op = nn_ops.conv1d
            strides = 1
        elif shape_length == 4:
            conv_op = nn_ops.conv2d
            strides = shape_length * [1]
        elif shape_length == 5:
            conv_op = nn_ops.conv3d
            strides = shape_length * [1]

        if len(args) == 1:
            inputs = args[0]
        else:
            inputs = array_ops.concat(axis=shape_length - 1, values=args)

        # Now the computation.
        # kernel = vs.get_variable(
        #     "kernel", filter_size + [total_arg_size_depth, num_features], dtype=dtype)
        kernel = tf.get_variable("kernel", filter_size + [total_arg_size_depth, num_features],
                                 dtype=dtype, initializer=self.weight_initializer)
        #res = tf.nn.conv2d(inputs, kernel, strides, padding='SAME')
        res = conv_op(inputs, kernel, strides, padding='SAME')

        '''
        if len(args) == 1:
            res = conv_op(args[0], kernel, strides, padding="SAME")
        else:
            res = conv_op(
                array_ops.concat(axis=shape_length - 1, values=args),
                kernel,
                strides,
                padding="SAME")
        '''
        if not bias:
            return res
        bias_term = vs.get_variable(
            "biases", [num_features],
            dtype=dtype,
            initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
        return res + bias_term

    def _gconv(self, inputs, state, dy_adj_mx, output_size, bias_start=0.0):
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
        #
        if dy_adj_mx is not None:
            # print('dy_adj_mx is right.')
            dy_adj_mx = tf.reshape(dy_adj_mx, (batch_size, self._num_nodes, -1))
        else:
            print('No dynamic flow input to generate dynamic adjacent matrix.')
            dy_adj_mx = None

        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        # x0 = x
        # x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        # x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        # x = tf.expand_dims(x0, axis=0)
        scope = tf.get_variable_scope()
        # dy_supports = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self._max_diffusion_step == 0:
                pass
            else:
                # get dynamic adj_mx
                # print(dy_adj_mx)
                dy_supports = self.get_supports(dy_adj_mx)
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

                    for k in range(2, self._max_diffusion_step + 1):
                        # x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x2 = 2 * tf.matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            # num_matrices = len(dy_supports) * self._max_diffusion_step + 1  # Adds for x itself.
            num_matrices = self._len_supports * self._max_diffusion_step + 1  # Adds for x itself.
            #
            x = tf.reshape(x, shape=[num_matrices, batch_size, self._num_nodes, input_size])
            x = tf.transpose(x, perm=[1, 2, 3, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)

        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])


    def calculate_random_walk_matrix(self, adj_mx):
        # adj_mx: [batch_size, num_nodes, num_nodes]
        # d = tf.sparse_tensor_to_dense(tf.sparse_reduce_sum(adj_mx, 1))
        d = tf.reduce_sum(adj_mx, -1)
        d_inv = tf.where(tf.greater(d, tf.zeros_like(d)) , tf.reciprocal(d), tf.zeros_like(d))
        d_mat_inv = tf.matrix_diag(d_inv)
        random_walk_mx = tf.matmul(d_mat_inv, adj_mx)
        return tf.cast(random_walk_mx, dtype=tf.float32)


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
