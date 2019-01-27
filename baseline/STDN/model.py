import tensorflow as tf
from utils import *
import os
from keras import backend as K, losses
import numpy as np
# import pandas as pd
# import argparse
import keras
from keras.models import Sequential, Model
from keras import activations
from keras.engine.topology import Layer, InputSpec
from keras.utils import conv_utils
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras import metrics
from keras.layers.normalization import BatchNormalization
from random import randint
import pickle


#batch_size = 64
mean_label = 0.0
label_max = 0
label_min = 0
max_epoch = 100
num_feature = 100
# seq_len = 8
cnn_flat_size = 128
hidden_dim = 64
threshold = 10.0
maxtruey = 0
mintruey = 0
#eps = 1e-5
eps = 1e-6
loss_lambda = 10.0
feature_len = 0
nbhd_size = 7
cnn_hidden_dim_first = 32
len_valid_id = 0
toponet_len = 32

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


def build_model(trainY, testY, trainX, testX, train_flow, test_flow,
                minMax,
                seq_len=6,
                batch_size=64,
                trainable=True,
                name="MODEL",
                model_path='./'):
    print(testX.shape)
    #
    nbhd_inputs = [Input(shape=(nbhd_size, nbhd_size, 2,), name="nbhd_volume_input_time_{0}".format(ts + 1)) for
                   ts in range(seq_len)]
    flow_inputs = [Input(shape=(nbhd_size, nbhd_size, 1,), name="flow_volume_input_time_{0}".format(ts + 1)) for
                   ts in range(seq_len)]

    # 1st level gate
    # nbhd cnn
    nbhd_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time0_{0}".format(ts + 1))(
        nbhd_inputs[ts]) for ts in range(seq_len)]
    nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time0_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in
                  range(seq_len)]
    # flow cnn
    flow_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="flow_convs_time0_{0}".format(ts + 1))(
        flow_inputs[ts]) for ts in range(seq_len)]
    flow_convs = [Activation("relu", name="flow_convs_activation_time0_{0}".format(ts + 1))(flow_convs[ts]) for ts in
                  range(seq_len)]
    # flow gate
    flow_gates = [Activation("sigmoid", name="flow_gate0_{0}".format(ts + 1))(flow_convs[ts]) for ts in
                  range(seq_len)]
    nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(seq_len)]

    # 2nd level gate
    nbhd_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time1_{0}".format(ts + 1))(
        nbhd_convs[ts]) for ts in range(seq_len)]
    nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time1_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in
                  range(seq_len)]
    flow_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="flow_convs_time1_{0}".format(ts + 1))(
        flow_inputs[ts]) for ts in range(seq_len)]
    flow_convs = [Activation("relu", name="flow_convs_activation_time1_{0}".format(ts + 1))(flow_convs[ts]) for ts in
                  range(seq_len)]
    flow_gates = [Activation("sigmoid", name="flow_gate1_{0}".format(ts + 1))(flow_convs[ts]) for ts in
                  range(seq_len)]
    nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(seq_len)]

    # 3rd level gate
    nbhd_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="nbhd_convs_time2_{0}".format(ts + 1))(
        nbhd_convs[ts]) for ts in range(seq_len)]
    nbhd_convs = [Activation("relu", name="nbhd_convs_activation_time2_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in
                  range(seq_len)]
    flow_convs = [Conv2D(filters=64, kernel_size=(3, 3), padding="same", name="flow_convs_time2_{0}".format(ts + 1))(
        flow_inputs[ts]) for ts in range(seq_len)]
    flow_convs = [Activation("relu", name="flow_convs_activation_time2_{0}".format(ts + 1))(flow_convs[ts]) for ts in
                  range(seq_len)]
    flow_gates = [Activation("sigmoid", name="flow_gate2_{0}".format(ts + 1))(flow_convs[ts]) for ts in
                  range(seq_len)]
    nbhd_convs = [keras.layers.Multiply()([nbhd_convs[ts], flow_gates[ts]]) for ts in range(seq_len)]

    # dense part
    nbhd_vecs = [Flatten(name="nbhd_flatten_time_{0}".format(ts + 1))(nbhd_convs[ts]) for ts in range(seq_len)]
    nbhd_vecs = [Dense(units=cnn_flat_size, name="nbhd_dense_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for ts in
                 range(seq_len)]
    nbhd_vecs = [Activation("relu", name="nbhd_dense_activation_time_{0}".format(ts + 1))(nbhd_vecs[ts]) for ts in
                 range(seq_len)]

    # concatenate
    nbhd_vec = Concatenate(axis=-1)(nbhd_vecs)
    nbhd_vec = Reshape(target_shape=(seq_len, cnn_flat_size))(nbhd_vec)
    # lstm
    lstm = LSTM(units=64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)(nbhd_vec)
    lstm_all = Dense(units=2)(lstm)
    pred_volume = Activation('tanh')(lstm_all)

    model = Model(inputs=[nbhd_inputs, flow_inputs],
                  outputs=pred_volume)
    sgd = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
    model.compile(loss=losses.mse, optimizer=sgd, metrics=[metrics.mse])
    earlyStopping = EarlyStopping(
        monitor='val_loss', patience=5, verbose=0, mode='min')
    # model.fit([trainimage, trainX, traintopo], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1,
    #           callbacks=[earlyStopping])

    fname_param = os.path.join(model_path, 'log', name)
    if not os.path.exists(fname_param):
        os.makedirs(fname_param)
    fname_param = os.path.join(fname_param, 'STDN.best.h5')
    # early_stopping = EarlyStopping(monitor='val_rmse', patience=5, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    if trainable:
        model.fit([trainX, train_flow], trainY, batch_size=batch_size, epochs=max_epoch, validation_split=0.1,
                  callbacks=[earlyStopping, model_checkpoint])
    else:
        model.load_weights(fname_param)
    # testLoss = model.evaluate([testimage, testtopo], testY)
    score = model.evaluate([trainX, train_flow], trainY, batch_size=batch_size, verbose=0)
    # print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], minMax.inverse(np.sqrt(score[1]))))
    print('Train score: %.6f se (norm): %.6f se (real): %.6f' %
          (score[0], score[1], minMax.inverse(minMax.inverse(score[1]))))

    score = model.evaluate(
        [testX, test_flow], testY, batch_size=batch_size, verbose=0)
    # print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
    #       (score[0], score[1], minMax.inverse(np.sqrt(score[1]))))
    print('Test score: %.6f se (norm): %.6f se (real): %.6f' %
          (score[0], score[1], minMax.inverse(minMax.inverse(score[1]))))
    # model.save('local_conv_lstm_total_embed.h5')

    prediction = model.predict([testX, test_flow], batch_size=batch_size, verbose=0)
    print(prediction.shape)
    test_mse = minMax.inverse(minMax.inverse(np.mean(np.square(prediction - testY))))
    print('test mse is %.6f, and rmse : %.6f' % (test_mse, np.sqrt(test_mse)))
    return prediction

    # return model
