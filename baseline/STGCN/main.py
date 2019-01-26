# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
parser.add_argument('-dataset', '--dataset', type=str, default='citibike', help='datasets: citibike')
parser.add_argument('-input_steps', '--input_steps', type=int, default=6, help='number of input steps')
parser.add_argument('-model_save', '--model_save', type=str, default='', help='path to save model')
parser.add_argument('-trained_adj_mx', '--trained_adj_mx', type=int, default=0, help='if training adjacent matrix')
parser.add_argument('-delta', '--delta', type=int, default=1e7, help='delta to calculate rescaled weighted matrix')
parser.add_argument('-epsilon', '--epsilon', type=float, default=0.8, help='epsilon to calculate rescaled weighted matrix')
#
parser.add_argument('--n_route', type=int, default=0)
parser.add_argument('--n_his', type=int, default=6)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#
data_folder = '../../datasets/' + args.dataset + '-data/data/'
output_folder = os.path.join('./data', args.dataset, 'model_save', args.model_save)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(f'Training configs: {args}')

# Data Preprocessing
if 'citibike' in args.dataset:
    split = [3672, 240, 480]
    data, n_route = data_gen_2(filename=[data_folder+'d_station.npy', data_folder+'p_station.npy'], split=split,
                               n_frame=args.input_steps+1)
    args.n_route = n_route

print(f'>> Loading dataset with Mean: {data.mean:.2f}, STD: {data.std:.2f}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
#blocks = [[2, 32, 64], [64, 32, 128]]
blocks = [[2, 32, 64]]

# Load wighted adjacency matrix W
if args.trained_adj_mx:
    L = tf.get_variable('weight_matrix', shape=(n, n), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer)
    Lk = cheb_poly_approx_tf(L, Ks, n)
    #W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_{n}.csv'))
else:
    # load customized graph weight matrix
    #W = weight_matrix(pjoin('./dataset', args.graph))
    w = np.load(data_folder + 'w.npy')
    W = get_rescaled_W(w, delta=args.delta, epsilon=args.epsilon)
    # Calculate graph kernel
    L = scaled_laplacian(W)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, Ks, n)

#tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))
tf.add_to_collection(name='graph_kernel', value=Lk)


if __name__ == '__main__':
    model_train(data, blocks, args)
    model_test(data, data.get_len('test'), n_his, n_pred, args.inf_mode)
