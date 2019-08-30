#from __future__ import division
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from progressbar import *
import tensorflow as tf
from utils import *
from op_utils import *

class ModelSolver(object):
    def __init__(self, model, data, val_data, preprocessing, **kwargs):
        self.model = model
        self.data = data
        self.val_data = val_data
        self.preprocessing = preprocessing
        self.cross_val = kwargs.pop('cross_val', False)
        self.cpt_ext = kwargs.pop('cpt_ext', False)
        self.weighted_loss = kwargs.pop('weighted_loss', False)
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 32)
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

    def train(self, test_data):
        x = self.data['x']
        y = self.data['y']
        x_val = self.val_data['x']
        y_val = self.val_data['y']
        # build graphs
        y_, loss = self.model.build_model()
        grad_loss = loss
        # train op
        with tf.variable_scope('optimizer', reuse=tf.AUTO_REUSE):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(grad_loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        gpu_options = tf.GPUOptions(allow_growth=True)
        tf.get_variable_scope().reuse_variables()

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()
            #summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)
            saver = tf.train.Saver(tf.global_variables())
            if self.pretrained_model is not None:
                print("Start training with pretrained model...")
                pretrained_model_path = os.path.join(self.model_path, self.pretrained_model)
                saver.restore(sess, pretrained_model_path)
            #curr_loss = 0
            start_t = time.time()
            for e in range(self.n_epochs):
                # =============================== train ===================================
                curr_loss = 0
                widgets = ['Train: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=len(x)).start()
                for i in range(len(x)):
                    pbar.update(i)
                    feed_dict = {self.model.x_c: np.array(x[i][0]), self.model.x_p: np.array(x[i][1]), self.model.x_t: np.array(x[i][2]),
                                self.model.x_ext: np.array(x[i][3]),
                                self.model.y: np.array(y[i])}
                    _, l = sess.run([train_op, loss], feed_dict)
                    curr_loss += l
                pbar.finish()
                # compute counts of all regions
                t_count = 0
                for c in range(len(y)):
                    t_count += np.prod(np.array(y[c]).shape)
                t_rmse = np.sqrt(curr_loss/t_count)
                #print("at epoch " + str(e) + ", train loss is " + str(curr_loss) + ' , ' + str(t_rmse) + ' , ' + str(self.preprocessing.real_loss(t_rmse)))
                # ================================= validate =================================
                print('validate for val data...')
                val_loss = 0
                widgets = ['Validation: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=len(x_val)).start()
                for i in range(len(y_val)):
                    pbar.update(i)
                    feed_dict = {self.model.x_c: np.array(x_val[i][0]), self.model.x_p: np.array(x_val[i][1]), self.model.x_t: np.array(x_val[i][2]),
                                self.model.x_ext: np.array(x_val[i][3]),
                                self.model.y: np.array(y_val[i])}
                    y_val_, l = sess.run([y_, loss], feed_dict=feed_dict)
                    val_loss += l
                pbar.finish()
                # y_val : [batches, batch_size, seq_length, row, col, channel]
                #print(np.array(y_val).shape)
                v_count = 0
                for v in range(len(y_val)):
                    v_count += np.prod(np.array(y_val[v]).shape)
                v_rmse = np.sqrt(val_loss/v_count)
                #print("at epoch " + str(e) + ", validate loss is " + str(val_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(v_rmse)))
                #print("elapsed time: ", time.time() - start_t)
                if (e+1)%self.save_every == 0:
                    save_name = os.path.join(self.model_path, 'model')
                    saver.save(sess, save_name, global_step=e+1)
                    print("model-%s saved." % (e+1))
                # ============================ for test data ===============================
                print('test for test data...')
                x_test = test_data['x']
                y_test = test_data['y']
                t_loss = 0
                y_pre_test = []
                widgets = ['Test: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=len(x_test)).start()
                for i in range(len(y_test)):
                    pbar.update(i)
                    feed_dict = {self.model.x_c: np.array(x_test[i][0]), self.model.x_p: np.array(x_test[i][1]),
                                 self.model.x_t: np.array(x_test[i][2]),
                                 self.model.x_ext: np.array(x_test[i][3]),
                                 self.model.y: np.array(y_test[i])}
                    y_pre_i, l = sess.run([y_, loss], feed_dict=feed_dict)
                    t_loss += l
                    y_pre_test.append(y_pre_i)
                pbar.finish()
                # y_val : [batches, batch_size, seq_length, row, col, channel]
                # print(np.array(y_test).shape)
                y_true = self.preprocessing.inverse_transform(np.array(y_test))
                y_prediction = self.preprocessing.inverse_transform(np.array(y_pre_test))
                rmse, mae, mape = RMSE(y_prediction, y_true), MAE(y_prediction, y_true), MAPE(y_prediction, y_true)
                text = 'at epoch %d, test loss is %.6f, test prediction rmse/mae/mape is %.6f/%.6f/%.6f \n' % (
                    e, t_loss, rmse, mae, mape)
                print("at epoch " + str(e) + ", train loss is " + str(curr_loss) + ' , ' + str(t_rmse) + ' , ' + str(self.preprocessing.real_loss(t_rmse)))
                print("at epoch " + str(e) + ", validate loss is " + str(val_loss) + ' , ' + str(rmse) + ' , ' + str(self.preprocessing.real_loss(v_rmse)))
                print(text)
            return np.array(y_prediction)

    def test(self, data):
        x = data['x']
        y = data['y']
        # build graphs
        y_, loss = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        y_pred_all = []

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            start_t = time.time()
            #y_pred_all = np.ndarray(y.shape)
            t_loss = 0
            for i in range(len(y)):
                feed_dict = {self.model.x_c: np.array(x[i][0]), self.model.x_p: np.array(x[i][1]), self.model.x_t: np.array(x[i][2]),
                             self.model.x_ext: np.array(x[i][3]),
                             self.model.y: np.array(y[i])}
                y_p, l = sess.run([y_, loss], feed_dict=feed_dict)
                t_loss += l
                y_pred_all.append(y_p)

            # y : [batches, batch_size, seq_length, row, col, channel]
            print(np.array(y).shape)
            y_true = self.preprocessing.inverse_transform(np.array(y_test))
            y_prediction = self.preprocessing.inverse_transform(np.array(y_pre_test))
            rmse, mae, mape = RMSE(y_prediction, y_true), MAE(y_prediction, y_true), MAPE(y_prediction, y_true)
            text = 'test loss is %.6f, test prediction rmse/mae/mape is %.6f/%.6f/%.6f \n' % (t_loss, rmse, mae, mape)
            print(text)
            return y_pred_all
            # if save_outputs:
            #     np.save('test_outputs.npy',y_pred_all)












