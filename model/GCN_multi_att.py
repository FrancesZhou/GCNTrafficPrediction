import sys
import numpy as np
import pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *
from tensorflow.contrib import legacy_seq2seq
from model.dcrnn_cell import DCGRUCell


class GCN_multi_att():
    def __init__(self, num_station, input_steps, output_steps,
                 att_inputs=[],
                 num_units=64,
                 max_diffusion_step=2,
                 dy_adj=1,
                 dy_filter=0,
                 f_adj_mx=None,
                 filter_type='dual_random_walk',
                 batch_size=32):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.att_inputs = att_inputs
        self.num_units = num_units
        self.max_diffusion_step = max_diffusion_step

        self.dy_adj = dy_adj
        self.dy_filter = dy_filter
        self.f_adj_mx = f_adj_mx
        self.filter_type = filter_type

        self.batch_size = batch_size

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()


        adj_mx = self.f_adj_mx
        self.cell = DCGRUCell(self.num_units, adj_mx=adj_mx, max_diffusion_step=self.max_diffusion_step,
                              num_nodes=self.num_station, num_proj=None,
                              input_dim=self.num_station*3, dy_adj=self.dy_adj,
                              dy_filter=self.dy_filter, output_dy_adj=False,
                              add_att_context=False,
                              activation=tf.nn.tanh,
                              reuse=tf.AUTO_REUSE, filter_type=self.filter_type)
        self.cell_with_projection = DCGRUCell(self.num_units, adj_mx=adj_mx, max_diffusion_step=max_diffusion_step,
                                              num_nodes=self.num_station, num_proj=3,
                                              input_dim=self.num_station*3, 
                                              dy_adj=self.dy_adj, dy_filter=0, output_dy_adj=False,
                                              add_att_context=True, att_inputs=self.att_inputs, att_hidden_dim=64,
                                              activation=tf.nn.tanh,
                                              reuse=tf.AUTO_REUSE, filter_type=self.filter_type)

        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 3])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.output_steps, self.num_station, 3])


    def build_easy_model(self, is_training=False):
        encoding_cells = self.cell
        decoding_cells = self.cell_with_projection
        x = tf.transpose(tf.reshape(self.x, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        f_all = tf.transpose(tf.reshape(self.f, (self.batch_size, self.input_steps, -1)), [1, 0, 2])
        # x: [input_steps, batch_size, num_station*2]
        # f_all: [input_steps, batch_size, num_station*num_station]
        inputs = tf.concat([x, f_all], axis=-1)
        inputs = tf.unstack(inputs, axis=0)
        #
        #l_decode = tf.transpose(tf.reshape(self.y, (self.batch_size, self.output_steps, -1)), [1,0,2])
        #f_decode = tf.tile(tf.expand_dims(f_all[-1], axis=0), (self.output_steps, 1, 1))
        #labels = tf.unstack(tf.concat([l_decode, f_decode], axis=-1), axis=0)
        #print(labels[0].get_shape().as_list())
        labels = tf.unstack(tf.reshape(self.y, (self.batch_size, self.output_steps, -1)), axis=1)
        #
        GO_SYMBOL = tf.zeros(shape=(self.batch_size, self.num_station*3+self.num_station*self.num_station))
        #GO_SYMBOL = tf.zeros(shape=(self.batch_size, self.num_station*3))

        with tf.variable_scope('GCN_SEQ'):

            labels.insert(0, GO_SYMBOL)

            # labels: (horizon+1, batch_size, num_nodes*output_dim]
            print('encode')
            # context_outputs, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)
            # context_outputs = tf.unstack(context_outputs, axis=0)
            # context_outputs.insert(0, tf.zeros(shape=(self.batch_size, self.num_station*self.num_station)))
            # print(len(context_outputs))

            _, enc_state = tf.contrib.rnn.static_rnn(encoding_cells, inputs, dtype=tf.float32)

            def _loop_function(prev, i):
                if is_training:
                    # Return either the model's prediction or the previous ground truth in training.
                    result = labels[i]
                else:
                    # Return the prediction of the model in testing.
                    result = prev
                #print(result.get_shape().as_list())
                #print(context_outputs[i].get_shape().as_list())
                #result = tf.concat([result, f_all[-1]], axis=-1)
                print(i)
                return result

            print('decode')
            outputs, final_state = legacy_seq2seq.rnn_decoder(labels, enc_state, decoding_cells,
                                                              loop_function=_loop_function)

        outputs = tf.stack(outputs[:-1], axis=1)
        self._outputs = tf.reshape(outputs, (self.batch_size, self.input_steps, self.num_station, -1), name='outputs')
        #loss = 2*tf.nn.l2_loss(self.y - self._outputs)
        self._outputs = tf.nn.relu(self._outputs)
        loss = 2*tf.nn.l2_loss(tf.log(self.y + 1) - tf.log(self._outputs + 1))
        return self._outputs, loss


    
    
    
    
    
    
    
