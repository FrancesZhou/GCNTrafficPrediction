# from __future__ import division
import numpy as np
from numpy import linalg as LA
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
    def __init__(self, model, train_data, test_data, preprocessing, **kwargs):
        self.model = model
        self.train_data = train_data
        #self.val_data = val_data
        self.test_data = test_data
        self.preprocessing = preprocessing
        self.cross_val = kwargs.pop('cross_val', False)
        self.cpt_ext = kwargs.pop('cpt_ext', False)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 1)
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
        #val_loader = self.val_data
        test_loader = self.test_data
        # build graphs
        y_, loss = self.model.build_model()
        #y_, test_loss = self.model.predict()

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            #grads = tf.gradients(loss, tf.trainable_variables())
            #grads_and_vars = list(zip(grads, tf.trainable_variables()))
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if grad is not None]
            train_op = optimizer.apply_gradients(capped_gvs)
            #train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

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
                saver.restore(sess, self.model_path + self.pretrained_model)
            #
            for e in range(self.n_epochs):
                # ========================== train ====================
                train_l2_loss = 0
                train_metric_loss = np.zeros(6, dtype=np.float32)
                #in_rmse, out_rmse, in_rmlse, out_rmlse, in_er, out_er
                train_loader.data_index = np.arange(train_loader.num_data-train_loader.input_steps-self.batch_size+1)
                num_train_batches = (train_loader.num_data - train_loader.input_steps - self.batch_size + 1)/self.batch_size
                print('number of training batches: %d' % num_train_batches)
                widgets = ['Train: ', Percentage(), ' ', Bar('-'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=num_train_batches).start()
                train_loader.reset_data()
                for i in xrange(num_train_batches):
                    # if i % self.show_batches == 0:
                    #     print 'train batch %d' % i
                    pbar.update(i)
                    #print i
                    #t1 = time.time()
                    x, f, ee, y, index = train_loader.next_batch_for_train(i*self.batch_size, (i+1)*self.batch_size)
                    #x, f, ee, y = train_loader.next_sample(i)
                    if x is None:
                        print 'invalid batch'
                        continue
                    #t2 = time.time()
                    #print 'load batch time: %s' % (t2-t1)
                    #print np.array(x).shape
                    feed_dict = {self.model.x: np.array(x),
                                 self.model.f: np.array(f),
                                 self.model.e: np.array(ee),
                                 self.model.y: np.array(y)
                                 }
                    _, l, y_out = sess.run([train_op, loss, y_], feed_dict)
                    y_out = np.round(self.preprocessing.inverse_transform(y_out[:, -1, :, :], index[:, -1]))
                    y = np.round(self.preprocessing.inverse_transform(y[:, -1, :, :], index[:, -1]))
                    metric_loss = get_loss_by_batch(y, y_out)
                    #t3 = time.time()
                    #print 'train batch time: %s' % (t3-t2)
                    train_l2_loss += l
                    train_metric_loss += metric_loss
                pbar.finish()
                # compute counts of all regions
                t_count = num_train_batches*self.batch_size*(train_loader.input_steps*train_loader.num_station*2)
                train_rmse = np.sqrt(train_l2_loss / t_count)
                train_metric_loss = train_metric_loss/(num_train_batches*self.batch_size)
                '''
                w_text = 'at epoch %d, train l2 loss is %.6f\n' \
                         'all in/out rmse is %.6f/%.6f\n' \
                         'all in/out rmlse is %.6f/%.6f\n' \
                         'all in/out er is %.6f/%.6f' % \
                         (e, train_rmse,
                          train_metric_loss[0], train_metric_loss[1],
                          train_metric_loss[2], train_metric_loss[3],
                          train_metric_loss[4], train_metric_loss[5])
                print w_text
                o_file.write(w_text)
                #'''
                if (e + 1) % self.save_every == 0:
                    save_name = self.model_path + 'model'
                    saver.save(sess, save_name, global_step=e + 1)
                    print "model-%s saved." % (e + 1)
                # ============================ for test data ===============================
                if e % 1 == 0:
                    print('test for test data...')
                    test_l2_loss = 0
                    test_metric_loss = np.zeros(6)
                    #num_test_batches = (test_loader.num_data - test_loader.input_steps - test_loader.output_steps + 1) / self.batch_size
                    num_test_batches = (test_loader.num_data - test_loader.input_steps - self.batch_size + 1) / self.batch_size
                    print('number of testing batches: %d' % num_test_batches)
                    widgets = ['Test: ', Percentage(), ' ', Bar('*'), ' ', ETA()]
                    pbar = ProgressBar(widgets=widgets, maxval=num_test_batches).start()
                    test_prediction = []
                    test_target = []
                    for i in xrange(num_test_batches):
                        # if i % self.show_batches == 0:
                        #     print 'test batch %d' % i
                        pbar.update(i)
                        #x, y, f, padding_len = test_loader.next_batch_for_test(i * self.batch_size, (i + 1) * self.batch_size)
                        x, f, ee, y, index = test_loader.next_batch_for_train(i * self.batch_size, (i + 1) * self.batch_size)
                        #x, f, ee, y = test_loader.next_sample(i)
                        feed_dict = {self.model.x: np.array(x),
                                     self.model.f: np.array(f),
                                     self.model.e: np.array(ee),
                                     self.model.y: np.array(y)
                                     }
                        y_out, l = sess.run([y_, loss], feed_dict)
                        y_out = np.round(self.preprocessing.inverse_transform(y_out[:,-1,:,:], index[:, -1]))
                        y = np.round(self.preprocessing.inverse_transform(y[:,-1,:,:], index[:, -1]))
                        test_prediction.append(y_out)
                        test_target.append(y)
                        metric_loss = get_loss_by_batch(y, y_out)
                        test_l2_loss += l
                        test_metric_loss += metric_loss
                    pbar.finish()
                    test_target = np.concatenate(np.array(test_target), axis=0)
                    test_prediction = np.concatenate(np.array(test_prediction), axis=0)
                    print test_target.shape
                    # compute counts of all regions
                    t_count = num_test_batches * self.batch_size * (test_loader.input_steps * test_loader.num_station * 2)
                    test_rmse = np.sqrt(test_l2_loss / t_count)
                    test_metric_loss = test_metric_loss / (num_test_batches * self.batch_size)
                    #w_text = 'at epoch %d, train l2 loss is %s\n' % (e, t_rmse)
                    #w_text += 'in/out rmse, rmlse, er is ' + str(test_metric_loss)
                    #'''
                    w_text = 'at epoch %d, train l2 loss is %.6f \t\t test all l2 loss is %.6f \n' \
                             'train in/out rmse is %.6f/%.6f \t test in/out rmse is %.6f/%.6f \n' \
                             'train in/out rmlse is %.6f/%.6f \t test in/out rmlse is %.6f/%.6f\n' \
                             'train in/out er is %.6f/%.6f \t\t test in/out er is %.6f/%.6f' % \
                             (e, train_rmse, test_rmse,
                              train_metric_loss[0], train_metric_loss[1], test_metric_loss[0], test_metric_loss[1],
                              train_metric_loss[2], train_metric_loss[3], test_metric_loss[2], test_metric_loss[3],
                              train_metric_loss[4], train_metric_loss[5], test_metric_loss[4], test_metric_loss[5])
                    #'''
                    print w_text
                    o_file.write(w_text)
            return np.array(test_target), np.array(test_prediction)



