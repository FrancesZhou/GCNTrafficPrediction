import os
import argparse
import numpy as np
import tensorflow as tf
#from gensim.models import Word2Vec
from model.AttGCN import AttGCN
from model.GCN import GCN
from model.ConvLSTM import ConvLSTM
from solver import ModelSolver
from preprocessing import *
from utils import *
from dataloader import *
# import scipy.io as sio


def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    parse.add_argument('-folder_name', '--folder_name', type=str, default='datasets/taxi-data/data/')
    parse.add_argument('-output_folder_name', '--output_folder_name', type=str, default='output/taxi-data/data/')
    parse.add_argument('-if_minus_mean', '--if_minus_mean', type=int, default=0,
                       help='use MinMaxNormalize01 or MinMaxNormalize01_minus_mean')
    # ---------- input/output settings -------
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6,
                       help='number of input steps')
    # ---------- model ----------
    # 1. dynamic_adj = 0 ==> ConvLSTM
    # 2. dynamic_adj = 1, dynamic_filter = 0 >>> f_kernel_size = kernel_size
    # 3. dynamic_adj = 1, dynamic_filter = 1 >>> f_kernel_size whatever.
    parse.add_argument('-model', '--model', type=str, default='ConvLSTM', help='model: ConvLSTM, GCN')
    parse.add_argument('-kernel_size', '--kernel_size', type=int, default=3, help='kernel size in convolutional operations')
    #parse.add_argument('-f_kernel_size', '--f_kernel_size', type=int, default=3, help='number of flow input channel')
    parse.add_argument('-dynamic_adj', '--dynamic_adj', type=int, default=1,
                       help='whether to use dynamic adjacent matrix for lower feature extraction layer')
    parse.add_argument('-dynamic_filter', '--dynamic_filter', type=int, default=1,
                       help='whether to use dynamic filter generate region-specific filter ')
    parse.add_argument('-model_save', '--model_save', type=str, default='gcn', help='folder name to save model')
    parse.add_argument('-pretrained_model', '--pretrained_model_path', type=str, default=None,
                       help='path to the pretrained model')
    # ---------- params for CNN ------------
    parse.add_argument('-num_filters', '--num_filters', type=int,
                       default=32, help='number of filters in CNN')
    parse.add_argument('-pooling_units', '--pooling_units', type=int,
                       default=64, help='number of pooling units')
    parse.add_argument('-dropout_keep_prob', '--dropout_keep_prob', type=float,
                       default=0.5, help='keep probability in dropout layer')
    # ---------- training parameters --------
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=20, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=8, help='batch size for training')
    parse.add_argument('-show_batches', '--show_batches', type=int,
                       default=100, help='show how many batches have been processed.')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.0002, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    # ---------- train or predict -------
    parse.add_argument('-train', '--train', type=int, default=1, help='whether to train')
    parse.add_argument('-test', '--test', type=int, default=0, help='if test')
    #
    args = parse.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('load train, test data...')
    # train: 20140101 - 20150430
    # validate: 20150501 - 20150531
    # test: 20150601 - 20150630
    split = [11640, 744, 720]
    data, train_data, val_data, test_data = load_npy_data(filename=[args.folder_name+'nyc_taxi_data.npy'], split=split)
    #
    map_size = data.shape[1:-1]
    input_dim = data.shape[-1]
    print('map_size: ' + str(map_size))
    print('input dim: %d' % input_dim)
    #
    if args.dynamic_adj:
        f_data, train_f_data, val_f_data, test_f_data = load_npy_data([args.folder_name + 'nyc_taxi_flow.npy'], split=split)
        print(len(f_data))
        if args.dynamic_filter == 0:
            indices = get_subarea_index(args.kernel_size, 7)
            #f_data = f_data[:,:,:, indices]
            train_f_data = train_f_data[:,:,:, indices]
            val_f_data = val_f_data[:,:,:, indices]
            test_f_data = test_f_data[:,:,:, indices]
        print('preprocess train/val/test flow data...')
        #f_preprocessing = StandardScaler()
        f_preprocessing = MinMaxNormalization01()
        f_preprocessing.fit(train_f_data)
        train_f_data = f_preprocessing.transform(train_f_data)
        val_f_data = f_preprocessing.transform(val_f_data)
        test_f_data = f_preprocessing.transform(test_f_data)
    else:
        train_f_data = np.zeros((split[0], map_size[0], map_size[1], args.kernel_size*args.kernel_size), dtype=np.float32)
        val_f_data = np.zeros((split[1], map_size[0], map_size[1], args.kernel_size*args.kernel_size), dtype=np.float32)
        test_f_data = np.zeros((split[2], map_size[0], map_size[1], args.kernel_size * args.kernel_size), dtype=np.float32)
    f_input_dim = train_f_data.shape[-1]
    print('preprocess train/val/test data...')
    #pre_process = MinMaxNormalization01_by_axis()
    if args.if_minus_mean:
        pre_process = MinMaxNormalization01_minus_mean()
    else:
        pre_process = MinMaxNormalization01()
        #pre_process = StandardScaler()
    pre_process.fit(train_data)
    train_data = pre_process.transform(train_data)
    val_data = pre_process.transform(val_data)
    test_data = pre_process.transform(test_data)
    #
    train_loader = DataLoader_map(train_data, train_f_data,
                                  args.input_steps)
    val_loader = DataLoader_map(val_data, val_f_data,
                                args.input_steps)
    test_loader = DataLoader_map(test_data, test_f_data,
                                 args.input_steps)

    if args.model == 'ConvLSTM':
        model = ConvLSTM(input_shape=[map_size[0], map_size[1], input_dim], input_steps=args.input_steps,
                         num_layers=3, num_units=32, kernel_shape=[args.kernel_size, args.kernel_size],
                         f_input_dim=f_input_dim,
                         dy_adj=args.dynamic_adj,
                         dy_filter=args.dynamic_filter,
                         batch_size=args.batch_size)
    #
    model_path = os.path.join(args.output_folder_name, 'model_save', args.model_save)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    #model_path = os.path.join(args.folder_name, 'model_save', args.model_save)
    solver = ModelSolver(model, train_loader, val_loader, test_loader, pre_process,
                         batch_size=args.batch_size,
                         show_batches=args.show_batches,
                         n_epochs=args.n_epochs,
                         pretrained_model=args.pretrained_model_path,
                         update_rule=args.update_rule,
                         learning_rate=args.learning_rate,
                         model_path=model_path,
                         )
    results_path = os.path.join(model_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if args.train:
        print('==================== begin training ======================')
        test_target, test_prediction = solver.train(os.path.join(model_path, 'out'))
        np.save(os.path.join(results_path, 'test_target.npy'), test_target)
        np.save(os.path.join(results_path, 'test_prediction.npy'), test_prediction)
    if args.test:
        print('==================== begin test ==========================')
        test_target, test_prediction = solver.test()
        np.save(os.path.join(results_path, 'test_target.npy'), test_target)
        np.save(os.path.join(results_path, 'test_prediction.npy'), test_prediction)


if __name__ == "__main__":
    main()
