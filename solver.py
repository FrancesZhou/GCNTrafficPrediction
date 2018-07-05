# from __future__ import division
import numpy as np
import time
import os
import math
from progressbar import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

sys.path.append('./util/')
from utils import *


class ModelSolver(object):
    def __init__(self, model, train_data, val_data, test_data, preprocessing, **kwargs):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.preprocessing = preprocessing
        self.cross_val = kwargs.pop('cross_val', False)
        self.cpt_ext = kwargs.pop('cpt_ext', False)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.show_batches = kwargs.pop('show_batches', 100)
        self.learning_rate = kwargs.pop('learning_rate', 0.000001)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.model_path = kwargs.pop('model_path', './model/')
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self, output_file_path=None):
        o_file = open(output_file_path, 'w')
        train_loader = self.train_data
        val_loader = self.val_data
        test_loader = self.test_data
        # build graphs
        train_loss = self.model.build_model()
        y_, test_loss = self.model.predict()

        # tf.get_variable_scope().reuse_variables()
        # y_ = self.model.build_sampler()

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(train_loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        tf.get_variable_scope().reuse_variables()
        # y_ = self.model.build_sampler()
        # summary op
        # tf.summary.scalar('batch_loss', train_loss)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
        # for grad, var in grads_and_vars:
        #     tf.summary.histogram(var.op.name + '/gradient', grad)

        # summary_op = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(tf.global_variables())
            if self.pretrained_model is not None:
                print "Start training with pretrained model..."
                saver.restore(sess, self.pretrained_model)
            #
            for e in range(self.n_epochs):
                # ========================== train ====================
                curr_loss = 0
                num_train_batches = int(math.ceil((train_loader.num_data-train_loader.input_steps) / self.batch_size))
                print('number of training batches: %d' % num_train_batches)
                widgets = ['Train: ', Percentage(), ' ', Bar('-'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=num_train_batches).start()
                #print('number of training batches: %d' % num_train_batches)
                train_loader.reset_data()
                for i in xrange(num_train_batches):
                    # if i % self.show_batches == 0:
                    #     print 'train batch %d' % i
                    pbar.update(i)
                    #print i
                    #t1 = time.time()
                    x, y, f = train_loader.next_batch_for_train(i*self.batch_size, (i+1)*self.batch_size)
                    if x is None:
                        continue
                    #t2 = time.time()
                    #print 'load batch time: %s' % (t2-t1)
                    feed_dict = {self.model.x: np.array(x),
                                 self.model.y_train: np.array(y),
                                 #self.model.f_train: np.array(f)
                                 }
                    _, l = sess.run([train_op, train_loss], feed_dict)
                    #t3 = time.time()
                    #print 'train batch time: %s' % (t3-t2)
                    curr_loss += l
                pbar.finish()
                # compute counts of all regions
                t_count = num_train_batches*self.batch_size*(train_loader.input_steps*train_loader.num_station*2)
                t_rmse = np.sqrt(curr_loss / t_count)
                # t_rmse = np.sqrt(curr_loss/(np.prod(np.array(y).shape)))
                w_text = "at epoch " + str(e) + ", train loss is " + str(curr_loss) + ' , ' + str(t_rmse) + ' , ' + str(
                    self.preprocessing.real_loss(t_rmse))
                print w_text
                o_file.write(w_text)
                # ========================== validate ===========================
                val_loss = 0
                y_pre = []
                num_val_batches = int(math.ceil((val_loader.num_data - val_loader.input_steps - val_loader.output_steps + 1) / self.batch_size))
                print('number of validation batches: %d' % num_val_batches)
                widgets = ['Validate: ', Percentage(), ' ', Bar('='), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=num_val_batches).start()
                #print('number of validation batches: %d' % num_val_batches)
                padding_count = 0
                for i in xrange(num_val_batches):
                    # if i % self.show_batches == 0:
                    #     print 'validate batch %d' % i
                    pbar.update(i)
                    x, y, f, padding_len = val_loader.next_batch_for_test(i * self.batch_size, (i + 1) * self.batch_size)
                    feed_dict = {self.model.x: np.array(x),
                                 self.model.y_test: np.array(y),
                                 #self.model.f_test: np.array(f)
                                 }
                    y_out, l = sess.run([y_, test_loss], feed_dict)
                    if padding_len > 0:
                        y_out = np.array(y_out[:-padding_len])
                        padding_count = np.prod(y_out.shape)
                    else:
                        y_pre.append(y_out)
                    val_loss += l
                pbar.finish()
                # compute counts of all regions
                # v_count = num_val_batches * self.batch_size * (val_loader.output_steps * val_loader.num_station * 2)
                # rmse = np.sqrt(val_loss / v_count)
                rmse = np.sqrt(val_loss/(np.prod(np.array(y_pre).shape) + padding_count))
                w_text = "at epoch " + str(e) + ", validate loss is " + str(val_loss) + ' , ' + str(rmse) + ' , ' + str(
                    self.preprocessing.real_loss(rmse))
                print w_text
                o_file.write(w_text)
                if (e + 1) % self.save_every == 0:
                    save_name = self.model_path + 'model'
                    saver.save(sess, save_name, global_step=e + 1)
                    print "model-%s saved." % (e + 1)
                # ============================ for test data ===============================
                if e == self.n_epochs-1:
                    print('test for test data...')
                    t_loss = 0
                    y_pre_test = []
                    num_test_batches = (test_loader.num_data - test_loader.input_steps - test_loader.output_steps + 1) / self.batch_size
                    print('number of testing batches: %d' % num_test_batches)
                    widgets = ['Test: ', Percentage(), ' ', Bar('*'), ' ', ETA()]
                    pbar = ProgressBar(widgets=widgets, maxval=num_test_batches).start()
                    #print('number of testing batches: %d' % num_test_batches)
                    padding_count = 0
                    for i in xrange(num_test_batches):
                        # if i % self.show_batches == 0:
                        #     print 'validate batch %d' % i
                        pbar.update(i)
                        x, y, f, padding_len = test_loader.next_batch_for_test(i * self.batch_size, (i + 1) * self.batch_size)
                        feed_dict = {self.model.x: np.array(x),
                                     self.model.y_test: np.array(y),
                                     #self.model.f_test: np.array(f)
                                     }
                        y_out, l = sess.run([y_, test_loss], feed_dict)
                        if padding_len > 0:
                            y_out = np.array(y_out[:-padding_len])
                            padding_count = np.prod(y_out.shape)
                        else:
                            y_pre_test.append(y_out)
                        t_loss += l
                    pbar.finish()
                    # compute counts of all regions
                    # t_count = num_test_batches * self.batch_size * (test_loader.output_steps * test_loader.num_station * 2)
                    # rmse = np.sqrt(t_loss / t_count)
                    rmse = np.sqrt(val_loss/(np.prod(np.array(y_pre_test).shape) + padding_count))
                    w_text = "at epoch " + str(e) + ", test loss is " + str(t_loss) + ' , ' + str(rmse) + ' , ' + str(
                        self.preprocessing.real_loss(rmse))
                    print w_text
                    o_file.write(w_text)



