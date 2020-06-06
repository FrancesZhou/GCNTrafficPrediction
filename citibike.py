import os
import argparse
import numpy as np
import tensorflow as tf
#from gensim.models import Word2Vec
from model.FC_LSTM import FC_LSTM
from model.FC_GRU import FC_GRU
from model.GCN import GCN
from model.Coupled_GCN import Coupled_GCN
from model.flow_GCN import flow_GCN
from solver import ModelSolver
from preprocessing import *
from utils import *
from dataloader import *
# import scipy.io as sio


def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    parse.add_argument('-folder_name', '--folder_name', type=str, default='datasets/citibike-data/data/')
    parse.add_argument('-output_folder_name', '--output_folder_name', type=str, default='output/citibike-data/data/')
    # ---------- input/output settings -------
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6,
                       help='number of input steps')
    # ---------- model ----------
    parse.add_argument('-model', '--model', type=str, default='GCN', help='model: DyST, GCN, AttGCN')
    parse.add_argument('-num_layers', '--num_layers', type=int, default=2, help='number of layers in model')
    parse.add_argument('-num_units', '--num_units', type=int, default=64, help='dim of hidden states')
    parse.add_argument('-trained_adj_mx', '--trained_adj_mx', type=int, default=0, help='if training adjacent matrix')
    parse.add_argument('-filter_type', '--filter_type', type=str, default='dual_random_walk', help='laplacian, random_walk, or dual_random_walk')
    parse.add_argument('-delta', '--delta', type=int, default=1e7, help='delta to calculate rescaled weighted matrix')
    parse.add_argument('-epsilon', '--epsilon', type=float, default=0.8, help='epsilon to calculate rescaled weighted matrix')
    parse.add_argument('-dy_adj', '--dy_adj', type=int, default=1,
                       help='whether to use dynamic adjacent matrix for lower feature extraction layer')
    parse.add_argument('-dy_filter', '--dy_filter', type=int, default=0,
                       help='whether to use dynamic filter generate region-specific filter ')
    #parse.add_argument('-att_dynamic_adj', '--att_dynamic_adj', type=int, default=1, help='whether to use dynamic adjacent matrix in attention parts')
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
    # train: 20140401 - 20140831
    # validate: 20140901 - 20140910
    # test: 20140911 - 20140930
    split = [3672, 240, 480]
    #split = [3912, 480]
    data, train_data, val_data, test_data = load_npy_data(
        filename=[args.folder_name+'d_station.npy', args.folder_name+'p_station.npy'], split=split)
    # data: [num, station_num, 2]
    #f_data, train_f_data, val_f_data, test_f_data = load_pkl_data(args.folder_name + 'f_data_list.pkl', split=split)
    f_data, train_f_data, val_f_data, test_f_data = load_npy_data(filename=[args.folder_name + 'citibike_flow_data.npy'], split=split)
    print(len(f_data))
    print('preprocess train/val/test flow data...')
    #f_preprocessing = StandardScaler()
    f_preprocessing = MinMaxNormalization01()
    f_preprocessing.fit(train_f_data)
    train_f_data = f_preprocessing.transform(train_f_data)
    if val_f_data is not None:
        val_f_data = f_preprocessing.transform(val_f_data)
    test_f_data = f_preprocessing.transform(test_f_data)
    print('preprocess train/val/test data...')
    pre_process = MinMaxNormalization01()
    #pre_process = StandardScaler()
    pre_process.fit(train_data)
    train_data = pre_process.transform(train_data)
    if val_data is not None:
        val_data = pre_process.transform(val_data)
    test_data = pre_process.transform(test_data)
    #
    num_station = data.shape[1]
    print('number of station: %d' % num_station)
    #
    train_loader = DataLoader_graph(train_data, train_f_data,
                              args.input_steps, flow_format='identity')
    if val_data is not None:
        val_loader = DataLoader_graph(val_data, val_f_data,
                                  args.input_steps, flow_format='identity')
    else:
        val_loader = None
    test_loader = DataLoader_graph(test_data, test_f_data,
                            args.input_steps, flow_format='identity')
    # f_adj_mx = None
    if os.path.isfile(args.folder_name + 'f_adj_mx.npy'):
        f_adj_mx = np.load(args.folder_name + 'f_adj_mx.npy')
    else:
        f_adj_mx = train_loader.get_flow_adj_mx()
        np.save(args.folder_name + 'f_adj_mx.npy', f_adj_mx)
    #
    #
    if args.filter_type == 'laplacian':
        w = np.load(args.folder_name + 'w.npy')
        # w = np.array(w, dtype=np.float32)
        W = get_rescaled_W(w, delta=args.delta, epsilon=args.epsilon)
        # Calculate graph kernel
        L = scaled_laplacian(W)
        #
        f_adj_mx = L

    if args.model == 'FC_LSTM':
        model = FC_LSTM(num_station, args.input_steps,
                        num_layers=args.num_layers, num_units=args.num_units,
                        batch_size=args.batch_size)
    if args.model == 'FC_GRU':
        model = FC_GRU(num_station, args.input_steps,
                        num_layers=args.num_layers, num_units=args.num_units,
                        batch_size=args.batch_size)
    if args.model == 'GCN':
        model = GCN(num_station, args.input_steps,
                    num_layers=args.num_layers, num_units=args.num_units,
                    dy_adj=args.dy_adj, dy_filter=args.dy_filter,
                    f_adj_mx=f_adj_mx, trained_adj_mx=args.trained_adj_mx,
                    filter_type=args.filter_type,
                    batch_size=args.batch_size)
    if args.model == 'flow_GCN':
        model = flow_GCN(num_station, args.input_steps,
                         num_layers=args.num_layers, num_units=args.num_units,
                         f_adj_mx=f_adj_mx, trained_adj_mx=args.trained_adj_mx,
                         filter_type=args.filter_type,
                         batch_size=args.batch_size)
    if args.model == 'Coupled_GCN':
        model = Coupled_GCN(num_station, args.input_steps,
                            num_layers=args.num_layers, num_units=args.num_units,
                            f_adj_mx=f_adj_mx, trained_adj_mx=args.trained_adj_mx,
                            filter_type=args.filter_type,
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
