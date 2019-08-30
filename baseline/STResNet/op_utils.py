import numpy as np
import pickle
import scipy.io as sio
import h5py
import time
import os

def RMSE(x_pre, x_true):
    x_pre = np.array(x_pre)
    x_true = np.array(x_true)
    return np.sqrt(np.mean(np.square(x_pre - x_true)))

def MAE(x_pre, x_true):
    x_pre = np.array(x_pre)
    x_true = np.array(x_true)
    return np.mean(np.abs(x_pre - x_true))

def MAPE(x_pre, x_true):
    x_pre = np.array(x_pre)
    x_true = np.array(x_true)
    return np.mean(np.abs((x_pre - x_true)/(x_true + 1)))
