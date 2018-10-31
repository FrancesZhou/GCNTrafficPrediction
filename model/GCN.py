import sys
import numpy as np
import pickle
import tensorflow as tf
sys.path.append('./util/')
from utils import *
from model.dcrnn_cell import DCGRUCell


class GCN():
    def __init__(self, num_station, input_steps, output_steps,
                 ext_dim=7,
                 num_units=64,
                 max_diffusion_step=2,
                 filter_type='dual_random_walk',
                 batch_size=32,
                 add_ext=0):
        self.num_station = num_station
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.ext_dim = ext_dim
        self.num_units = num_units
        self.max_diffusion_step = max_diffusion_step
        self.filter_type = filter_type

        self.batch_size = batch_size
        self.add_ext = add_ext

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer()

        
        self.cell = DCGRUCell(self.num_units, self.max_diffusion_step, self.num_station, reuse=tf.AUTO_REUSE, filter_type=self.filter_type)
        self.cell_with_projection = DCGRUCell(self.num_units, max_diffusion_step=max_diffusion_step, num_nodes=self.num_station, num_proj=2, filter_type=self.filter_type)


        self.x = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])
        self.f = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, self.num_station])
        self.e = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.ext_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.input_steps, self.num_station, 2])


    def build_model(self):
        x = tf.unstack(tf.reshape(self.x, (self.batch_size, self.input_steps, self.num_station*2)), axis=1)
        f_all = tf.unstack(tf.reshape(self.f, (self.batch_size, self.input_steps, self.num_station*self.num_station)), axis=1)

        e_all = tf.transpose(self.e, [1, 0, 2])
        y = self.y
        hidden_state = tf.zeros([self.batch_size, self.num_station*self.num_units])
        #current_state = tf.zeros([self.batch_size, self.num_station*self.num_unists])
        #state = hidden_state, current_state
        state_1 = hidden_state
        state_2 = hidden_state
        y_ = []
        for i in range(self.input_steps):
            # for each step
            #print(i)
            f = f_all[i]
            current_step_batch = x[i]
            with tf.variable_scope('dcrnn', reuse=tf.AUTO_REUSE):
                output_1, state_1 = self.cell(tf.reshape(current_step_batch, (self.batch_size, -1)), f, state_1)
            with tf.variable_scope('output', reuse=tf.AUTO_REUSE):
                output_2, state_2 = self.cell_with_projection(tf.reshape(output_1, (self.batch_size, -1)), f, state_2)
            # output: [batch_size, state_size]
            output_2 = tf.reshape(output_2, (self.batch_size, self.num_station, -1))
            #
            '''
            if self.add_ext:
                e = e_all[i]
                out_3 = tf.matmul(e, self.w_e_out)
                in_3 = tf.matmul(e, self.w_e_in)
            else:
                out_3 = tf.constant(0.0, dtype=tf.float32, shape=[self.batch_size, self.num_station])
                in_3 = tf.constant(0.0, dtype=tf.float32, shape=[self.batch_size, self.num_station])
            next_out = out_1 + out_2 + out_3
            # check-in
            #in_1 = tf.squeeze(tf.matmul(tf.multiply(f_out_gate, self.w_2), tf.expand_dims(x_out, -1)))
            in_1 = tf.squeeze(tf.matmul(tf.multiply(f_in_gate, self.w_2), tf.expand_dims(x_out, -1)))
            in_2 = tf.reduce_sum(tf.multiply(cxt_in, self.w_h_in), axis=-1) + tf.matmul(output, self.w_t_in)
            next_in = in_1 + in_2 + in_3
            next_output = tf.concat((tf.expand_dims(next_in, -1), tf.expand_dims(next_out, -1)), -1)
            y_.append(next_output)
            '''
            y_.append(output_2)
        y_ = tf.stack(y_)
        y_ = tf.transpose(y_, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(y-y_)
        return y_, loss


    def build_easy_model(self):
        x = tf.unstack(tf.reshape(self.x, (self.batch_size, self.input_steps, self.num_station*2)), axis=1)
        #print(len(x))
        f_all = tf.unstack(tf.reshape(self.f, (self.batch_size, self.input_steps, self.num_station*self.num_station)), axis=1)
        #print(len(f_all))
        inputs = list(zip(*(x, f_all)))
        self.cells = tf.contrib.rnn.MultiRNNCell([self.cell, self.cell_with_projection], state_is_tuple=True)
        outputs, _ = tf.contrib.rnn.static_rnn(self.cells, inputs, dtype=tf.float32)
        outputs = tf.reshape(outputs, (self.input_steps, self.batch_size, self.num_station, -1))
        outputs = tf.transpose(outputs, [1, 0, 2, 3])
        loss = 2*tf.nn.l2_loss(self.y - outputs)
        return outputs, loss
    
    
    
    
    
    
    
    
