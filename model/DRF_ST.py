import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('./util/')
from utils import *
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from model.modules import *
from model.dcrnn_cell import DCGRUCell
from model.coupled_convgru_cell import Coupled_Conv2DGRUCell


class DRF_ST():
    def __init__(self, input_shape=[20,10,2], adj_mx=None,
                 structure='grid', use_spatial=1, use_flow=1, trained_adj_mx=0,
                 input_steps=6,
                 num_layers=2, num_units=64, num_heads=8,
                 kernel_shape=[3,3], max_diffusion_step=2, filter_type='dual_random_walk',
                 dropout_rate=0.3, batch_size=32):
        self.input_shape = input_shape
        self.adj_mx = adj_mx
        #
        self.structure = structure
        self.use_spatial = use_spatial
        self.use_flow = use_flow
        #
        self.input_steps = input_steps
        #
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_heads = num_heads
        #
        self.kernel_shape = kernel_shape
        self._max_diffusion_step = max_diffusion_step
        self.filter_type = filter_type
        #
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self._supports = []
        #
        self._input_dim = self.input_shape[-1]
        self._num_nodes = np.prod(self.input_shape[:-1])
        '''
        if self.structure == 'grid':
            self._num_nodes = np.prod(self.input_shape[:-1])
        elif self.structure == 'graph':
            self._num_nodes = self.num_station
        '''
        if trained_adj_mx:
            with tf.variable_scope('trained_adj_mx', reuse=tf.AUTO_REUSE):
                adj_mx = tf.get_variable('adj_mx', [self._num_nodes, self._num_nodes], dtype=tf.float32,
                                         initializer=self.weight_initializer)
        if adj_mx is not None:
            # for fixed adjacent matrix
            if self.filter_type == 'laplacian':
                self._supports.append(tf.convert_to_tensor(adj_mx, dtype=tf.float32))
            elif self.filter_type == "random_walk":
                self._supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx)))
            elif self.filter_type == "dual_random_walk":
                self._supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx)))
                self._supports.append(tf.transpose(self.calculate_random_walk_matrix(tf.transpose(adj_mx))))

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self._num_nodes, self._input_dim])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self._num_nodes, self._num_nodes])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self._num_nodes, self._input_dim])

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def build_easy_model(self, training):
        x, f = self.x, self.f
        for i in range(self.num_layers):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                # space modeling
                x_space = self.space_modeling(x, f)
                # time modeling
                x_time = self.time_modeling(x_space, training)
                # feed forward
                #x_time = tf.reshape(x_time, (self.batch_size*self.input_steps*self._num_nodes, -1))
                with tf.variable_scope('feedforward', reuse=tf.AUTO_REUSE):
                    outputs = tf.layers.dense(x_time, self.num_units, activation=tf.nn.relu)
                    #outputs += x_time
                    #outputs = ln(outputs)
            x = outputs

        # projection
        outputs = tf.layers.dense(x, units=self._input_dim, activation=None, kernel_initializer=self.weight_initializer)
        #
        #outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.input_shape[0], self.input_shape[1], -1))
        loss = 2 * tf.nn.l2_loss(self.y - outputs)
        return outputs, loss


    def time_modeling(self, x, training):
        # x: [batch_size, input_steps, num_nodes, -1]
        x = tf.reshape(x, (self.batch_size, self.input_steps, self._num_nodes, -1))
        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (self.batch_size*self._num_nodes, self.input_steps, -1))
        x_temporal = multihead_attention(queries=x,
                                  keys=x,
                                  values=x,
                                  num_heads=self.num_heads,
                                  dropout_rate=self.dropout_rate,
                                  training=training,
                                  causality=False,
                                  scope="self_attention")
        x_temporal = tf.transpose(tf.reshape(x_temporal, (self.batch_size, self._num_nodes, self.input_steps, -1)), (0, 2, 1, 3))
        return x_temporal

    def space_modeling(self, x, f, bias_start=0.0):
        # x: [batch_size, input_steps, num_nodes, -1]
        if self.use_spatial:
            if self.structure == 'grid':
                with tf.variable_scope("spatial_grid", reuse=tf.AUTO_REUSE):
                    spatial_inputs_4d = tf.reshape(x, (self.batch_size*self.input_steps, self.input_shape[0], self.input_shape[1], -1))
                    x_spatial = self._conv(args=[spatial_inputs_4d],
                                        filter_size=self.kernel_shape,
                                        num_features=self.num_units,
                                        bias=False, bias_start=0)
            elif self.structure == 'graph':
                with tf.variable_scope("spatial_graph", reuse=tf.AUTO_REUSE):
                    spatial_inputs_3d = tf.reshape(x, (self.batch_size*self.input_steps, self._num_nodes, -1))
                    x_spatial = self._gconv(spatial_inputs_3d, None, self.num_units, False)
            x_spatial = tf.reshape(x_spatial, (self.batch_size, self.input_steps, self._num_nodes, self.num_units))
        else:
            x_spatial = tf.zeros((self.batch_size, self.input_steps, self._num_nodes, self.num_units))
        ##########################
        if self.use_flow:
            with tf.variable_scope("flow_modeling", reuse=tf.AUTO_REUSE):
                flow_inputs_3d = tf.reshape(x, (self.batch_size*self.input_steps, self._num_nodes, -1))
                flow_f_3d = tf.reshape(f, (self.batch_size*self.input_steps, self._num_nodes, self._num_nodes))
                x_flow = self._gconv(flow_inputs_3d, flow_f_3d, self.num_units, False)
                x_flow = tf.reshape(x_flow, (self.batch_size, self.input_steps, self._num_nodes, self.num_units))
        else:
            x_flow = tf.zeros((self.batch_size, self.input_steps, self._num_nodes, self.num_units))
        ##########################
        with tf.variable_scope("biases", reuse=tf.AUTO_REUSE):
            biases = tf.get_variable("biases", [1, 1, self._num_nodes, self.num_units], dtype=tf.float32,
                                         initializer=tf.constant_initializer(bias_start, dtype=tf.float32))

        return x_spatial + x_flow + biases

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
        # TODO: why does it need a transpose for the generated random_walk_matrix?
        if filter_type == "random_walk":
            supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx), (0, 2, 1)))
        elif filter_type == "dual_random_walk":
            supports.append(tf.transpose(self.calculate_random_walk_matrix(adj_mx), (0, 2, 1)))
            supports.append(
                tf.transpose(self.calculate_random_walk_matrix(tf.transpose(adj_mx, (0, 2, 1))), (0, 2, 1)))
        else:
            return None
        '''
        dy_supports = []
        for support in supports:
            dy_supports.append(self._build_sparse_matrix(support))
        return dy_supports
        '''
        return supports

    def _gconv(self, inputs, dy_adj_mx, output_size, bias, bias_start=0.0):
        """Graph convolution between input and the graph matrix.
        :param inputs: [batch_size, num_nodes, num_units]
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.get_shape()[2].value
        dtype = inputs.dtype

        if dy_adj_mx is None:
            if len(self._supports) == 0:
                print('No adjacent matrix is provided for spatial correlation modeling in graph-structure data.')

        x = inputs
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
                if dy_adj_mx is None:
                    dy_supports = self._supports
                    # print(dy_supports)
                    x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
                    x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
                    x = tf.expand_dims(x0, axis=0)
                else:
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

            num_matrices = len(dy_supports) * self._max_diffusion_step + 1  # Adds for x itself.
            #num_matrices = self._len_supports * self._max_diffusion_step + 1  # Adds for x itself.
            if dy_adj_mx is None:
                x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
                x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            else:
                x = tf.reshape(x, shape=[num_matrices, batch_size, self._num_nodes, input_size])
                x = tf.transpose(x, perm=[1, 2, 3, 0])  # (batch_size, num_nodes, input_size, order)

            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)
            if not bias:
                biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                         initializer=tf.constant_initializer(bias_start, dtype=dtype))
                x = tf.nn.bias_add(x, biases)

        # (batch_size * num_node, output_size)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])

    def _conv(self, args, filter_size, num_features, bias, bias_start=0.0):
        # Calculate the total size of arguments on dimension 1.
        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        shape_length = len(shapes[0])
        for shape in shapes:
            total_arg_size_depth += shape[-1]
        dtype = [a.dtype for a in args][0]
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
        res = tf.nn.conv2d(inputs, kernel, strides, padding='SAME')
        #res = nn_ops.conv2d(inputs, kernel, strides, padding='SAME')
        #
        if not bias:
            return res
        bias_term = vs.get_variable(
            "biases", [num_features],
            dtype=dtype,
            initializer=self.constant_initializer(bias_start, dtype=dtype))
        return res + bias_term










