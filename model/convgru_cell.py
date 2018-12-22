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


class Dy_Conv2DGRUCell(rnn_cell_impl.RNNCell):
    """Convolutional LSTM recurrent network cell.

    https://arxiv.org/pdf/1506.04214v1.pdf
    """

    def __init__(self,
                 input_shape,
                 output_channels,
                 kernel_shape,
                 input_dim=None, dy_adj=0, dy_filter=0, output_dy_adj=0,
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=1.0,
                 initializers=None,
                 name="conv_gru_cell"):
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
        super(Dy_Conv2DGRUCell, self).__init__(name=name)

        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape

        self._input_dim = input_dim
        self.dy_adj = dy_adj
        self.dy_filter = dy_filter
        self.output_dy_adj = output_dy_adj

        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._skip_connection = skip_connection

        self._total_output_channels = output_channels
        if self._skip_connection:
            self._total_output_channels += self._input_shape[-1]

#         state_size = tensor_shape.TensorShape(
#             self._input_shape[:-1] + [self._output_channels])
        #self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
        self._output_size = tensor_shape.TensorShape(
            self._input_shape[:-1] + [self._total_output_channels])
        
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return tensor_shape.TensorShape(self._input_shape[:-1] + [self._output_channels])
        #return self._state_size

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
        hidden = state
        new_hidden = self._conv(args=[inputs, hidden], filter_size=self._kernel_shape,
                           num_features=3 * self._output_channels, bias=self._use_bias, bias_start=0,
                           dy_f=dy_f)
        gates = array_ops.split(
            value=new_hidden, num_or_size_splits=3, axis=3)

        r, u, c = gates
        output = new_state = u * state + (1 - u) * c
        #input_gate, new_input, forget_gate, output_gate = gates
        #new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
        #new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
        #output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

        if self._skip_connection:
            output = array_ops.concat([output, inputs], axis=-1)
        #new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
        if self.output_dy_adj>0:
            print(output.get_shape().as_list())
            print(dy_f.get_shape().as_list())
            output = tf.concat([output, dy_f], axis=-1)
        return output, new_state

    def _conv(self, args, filter_size, num_features, bias, bias_start=0.0,
              dy_f=None):
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
        # dynamic flow
        if self.dy_adj > 0:
            if dy_f is None:
                print('No dynamic flow input to generate dynamic convolutional weights.')

        # Now the computation.
        # kernel = vs.get_variable(
        #     "kernel", filter_size + [total_arg_size_depth, num_features], dtype=dtype)

        if self.dy_adj > 0:
            filter_local_expand = tf.reshape(
                tf.eye(total_arg_size_depth * filter_size[0] * filter_size[1], dtype=tf.float32),
                (total_arg_size_depth * filter_size[0] * filter_size[1], total_arg_size_depth, filter_size[0],
                 filter_size[1]))
            filter_local_expand = tf.transpose(filter_local_expand, (2, 3, 1, 0))
            # filter_local_expand: [s, s, input_channel, input_channel*s*s]
            input_local_expanded = tf.nn.conv2d(inputs, filter_local_expand, strides, padding='SAME')
            # input_local_expanded: [batch_size, row, col, input_channel*s*s]
            input_local_all = tf.tile(tf.expand_dims(input_local_expanded, 3), (1, 1, 1, num_features, 1))
            # input_local_all: [batch_size, row, col, output_channel, input_channel*s*s]
            input_local_all = tf.reshape(input_local_all,
                                         (-1, num_features, total_arg_size_depth * filter_size[0] * filter_size[1]))
            # input_local_all: [batch_size*row*col, output_channel, input_channel*s*s]
            #####
            kernel = vs.get_variable(
                "kernel", filter_size + [num_features, total_arg_size_depth], dtype=dtype)
            reshape_kernel = tf.reshape(kernel, (filter_size[0], filter_size[1], -1))
            #
            if self.dy_filter > 0:
                f_input_dim = dy_f.get_shape().as_list()[-1]
                dy_filter_kernel = tf.get_variable("dy_filter_kernel", filter_size + [f_input_dim, filter_size[0]*filter_size[1]], dtype=dtype, initializer=self.weight_initializer)
                dy_gen_f = tf.nn.conv2d(dy_f, dy_filter_kernel, strides=strides, padding='SAME')
                reshape_dy_f = tf.reshape(dy_gen_f, (-1, filter_size[0], filter_size[1]))
            else:
                dy_f = tf.reshape(dy_f,
                                  (-1, self._input_shape[0], self._input_shape[1], filter_size[0], filter_size[1]))
                reshape_dy_f = tf.reshape(dy_f, (-1, filter_size[0], filter_size[1]))
            #
            dy_kernel = tf.expand_dims(reshape_dy_f, -1) * tf.expand_dims(reshape_kernel, 0)
            # dy_kernel: [batch_size*row*col, s, s, output_channel*input_channel]
            dy_kernel = tf.reshape(dy_kernel,
                                   (-1, filter_size[0], filter_size[1], num_features, total_arg_size_depth))
            dy_kernel = tf.transpose(dy_kernel, (0, 3, 4, 1, 2))
            dy_kernel = tf.reshape(dy_kernel,
                                   (-1, num_features, total_arg_size_depth * filter_size[0] * filter_size[1]))
            # dy_kernel: [batch_size*row*col, output_channel, input_channel*s*s]
            res = tf.reshape(tf.reduce_sum(input_local_all * dy_kernel, -1),
                             (-1, self._input_shape[0], self._input_shape[1], num_features))
        else:
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
