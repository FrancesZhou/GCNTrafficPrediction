import tensorflow as tf
from utils import *
import os
from keras import backend as K, losses
import numpy as np
import pandas as pd
import argparse
from keras.models import Sequential, Model
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras.layers import LSTM, InputLayer, Dense, Input, Flatten, concatenate, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras import metrics
from keras.layers.normalization import BatchNormalization
from random import randint
import pickle

# project_path = "/home/pengshunfeng/project/DMVSTNet"
# project_path = "E:/pyCharm/DMVSTNet"

batch_size = 64
mean_label = 0.0
label_max = 0
label_min = 0
max_epoch = 100
num_feature = 100
# seq_len = 8
hidden_dim = 512
threshold = 10.0
maxtruey = 0
mintruey = 0
eps = 1e-5
loss_lambda = 10.0
feature_len = 0
local_image_size = 9
cnn_hidden_dim_first = 32
len_valid_id = 0
toponet_len = 32

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


class Local_Seq_Conv(Layer):

    def __init__(self, output_dim, seq_len, feature_size, kernel_size, activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', padding='same', strides=(1, 1), **kwargs):
        super(Local_Seq_Conv, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.bias_initializer = bias_initializer
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def build(self, input_shape):
        batch_size = input_shape[0]
        self.kernel = []
        self.bias = []
        for eachlen in range(self.seq_len):
            self.kernel += [self.add_weight(shape=self.kernel_size,
                                            initializer=self.kernel_initializer,
                                            trainable=True, name='kernel_{0}'.format(eachlen))]

            self.bias += [self.add_weight(shape=(self.kernel_size[-1],),
                                          initializer=self.bias_initializer,
                                          trainable=True, name='bias_{0}'.format(eachlen))]
        self.build = True

    def call(self, inputs):
        output = []
        for eachlen in range(self.seq_len):

            tmp = K.bias_add(K.conv2d(inputs[:, eachlen, :, :, :], self.kernel[eachlen],
                                      strides=self.strides, padding=self.padding), self.bias[eachlen])

            if self.activation is not None:
                output += [self.activation(tmp)]

        output = tf.stack(output, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3], self.output_dim)

def squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))

def build_model(trainY, testY, trainimage, testimage, traintopo, testtopo,
                feature_len,
                minMax,
                seq_len=8,
                trainable=True,
                name="MODEL",
                model_path='./'):
    print(testimage.shape)
    # X_train, Y_train, X_test, Y_test = Featureset_get()
    image_input = Input(shape=(seq_len, local_image_size,
                               local_image_size, None), name='cnn_input')
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, 1, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(image_input)
    spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len,
                             kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu',
                             kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same',
                             strides=(1, 1))(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    # spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len, kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    # spatial = BatchNormalization()(spatial)
    # spatial = Local_Seq_Pooling(seq_len=seq_len)(spatial)
    # spatial = Local_Seq_Conv(output_dim=cnn_hidden_dim_first, seq_len=seq_len, feature_size=feature_len, kernel_size=(3, 3, cnn_hidden_dim_first, cnn_hidden_dim_first), activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', padding='same', strides=(1, 1))(spatial)
    spatial = Flatten()(spatial)
    spatial = Reshape(target_shape=(seq_len, -1))(spatial)
    spatial_out = Dense(units=64, activation='relu')(spatial)

    # lstm_input = Input(shape=(seq_len, feature_len),
    #                    dtype='float32', name='lstm_input')
    #
    # x = concatenate([lstm_input, spatial_out], axis=-1)
    x = spatial_out
    # lstm_out = Dense(units=128, activation=relu)(x)
    lstm_out = LSTM(units=hidden_dim, return_sequences=False, dropout=0)(x)

    topo_input = Input(shape=(toponet_len,), dtype='float32', name='topo_input')
    topo_emb = Dense(units=6, activation='tanh')(topo_input)
    static_dynamic_concate = concatenate([lstm_out, topo_emb], axis=-1)

    res = Dense(units=1, activation='sigmoid')(static_dynamic_concate)
    # model = Model(inputs=[image_input, lstm_input, topo_input],
    #               outputs=res)
    model = Model(inputs=[image_input, topo_input],
                  outputs=res)
    sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    #model.compile(loss=losses.mse, optimizer=sgd, metrics=[metrics.mse])
    model.compile(loss=squared_error, optimizer=sgd, metrics=[metrics.mse])
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='min')
    # model.fit([trainimage, trainX, traintopo], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1,
    #           callbacks=[earlyStopping])

    fname_param = os.path.join(model_path, 'log', name)
    if not os.path.exists(fname_param):
        os.makedirs(fname_param)
    fname_param = os.path.join(fname_param, 'DMVSTNet.best.h5')
    # early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    if trainable:
        model.fit([trainimage, traintopo], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1,
                  callbacks=[earlyStopping, model_checkpoint])
    else:
        model.load_weights(fname_param)
    # testLoss = model.evaluate([testimage, testtopo], testY)
    score = model.evaluate([trainimage, traintopo], trainY, batch_size=batch_size, verbose=0)
    # print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], minMax.inverse(np.sqrt(score[1]))))
    print('Train score: %.6f se (norm): %.6f se (real): %.6f' %
          (score[0], score[1], minMax.inverse(minMax.inverse(score[1]))))

    score = model.evaluate(
        [testimage, testtopo], testY, batch_size=batch_size, verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], minMax.inverse(np.sqrt(score[1]))))
    print('Test score: %.6f se (norm): %.6f se (real): %.6f' %
          (score[0], score[1], minMax.inverse(minMax.inverse(score[1]))))
    # model.save('local_conv_lstm_total_embed.h5')

    prediction = model.predict([testimage, testtopo], batch_size=batch_size, verbose=0)
    print(prediction.shape)
    test_mse = minMax.inverse(minMax.inverse(np.mean(np.square(prediction - testY))))
    print('test mse is %.6f, and rmse : %.6f' % (test_mse, np.sqrt(test_mse)))
    return prediction

    # return model
