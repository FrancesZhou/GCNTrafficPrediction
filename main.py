import os
import argparse
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from model.DyST import DyST
from model.DyST2 import DyST2
from model.DyST3 import DyST3
from solver import ModelSolver
from preprocessing import *
from utils import *
from dataloader import *
import scipy.io as sio


def main():
    parse = argparse.ArgumentParser()
    # ---------- environment setting: which gpu -------
    parse.add_argument('-gpu', '--gpu', type=str, default='0', help='which gpu to use: 0 or 1')
    parse.add_argument('-folder_name', '--folder_name', type=str, default='datasets/citibike-data/data/')
    parse.add_argument('-if_minus_mean', '--if_minus_mean', type=int, default=0,
                       help='use MinMaxNormalize01 or MinMaxNormalize01_minus_mean')
    # ---------- input/output settings -------
    parse.add_argument('-input_steps', '--input_steps', type=int, default=6,
                       help='number of input steps')
    parse.add_argument('-output_steps', '--output_steps', type=int, default=1,
                       help='number of output steps')
    # ---------- station embeddings --------
    parse.add_argument('-pretrained_embeddings', '--pretrained_embeddings', type=int, default=1,
                       help='whether to use pretrained embeddings')
    parse.add_argument('-embedding_size', '--embedding_size', type=int, default=100,
                       help='dim of embedding')
    # ---------- model ----------
    #parse.add_argument('-model', '--model', type=str, default='DyST', help='model: NN, LSTM, biLSTM, CNN')
    parse.add_argument('-dynamic_context', '--dynamic_context', type=int, default=1, help='whether to add dynamic_context part')
    parse.add_argument('-dynamic_spatial', '--dynamic_spatial', type=int, default=1, help='whether to add dynamic_spatial part')
    parse.add_argument('-add_ext', '--add_ext', type=int, default=1, help='whether to add external factors')
    parse.add_argument('-model_save', '--model_save', type=str, default='', help='folder name to save model')
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
    parse.add_argument('-n_epochs', '--n_epochs', type=int, default=50, help='number of epochs')
    parse.add_argument('-batch_size', '--batch_size', type=int, default=8, help='batch size for training')
    parse.add_argument('-show_batches', '--show_batches', type=int,
                       default=100, help='show how many batches have been processed.')
    parse.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='learning rate')
    parse.add_argument('-update_rule', '--update_rule', type=str, default='adam', help='update rule')
    # ------ train or predict -------
    parse.add_argument('-train', '--train', type=int, default=1, help='whether to train')
    parse.add_argument('-test', '--test', type=int, default=0, help='if test')
    #
    parse.add_argument('-pretrain', '--pretrain', type=int, default=0, help='whether to pretrain')
    parse.add_argument('-partial_pretrain', '--partial_pretrain', type=int, default=0, help='whether to load pretrained vars')
    args = parse.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('load train, test data...')
    # train: 20140401 - 20140910
    # test: 20140911 - 20140930
    split = [3912, 480]
    data, train_data, test_data, _ = load_npy_data(
        filename=[args.folder_name+'d_station.npy', args.folder_name+'p_station.npy'], split=split)
    # data: [num, station_num, 2]
    f_data, train_f_data, test_f_data, _ = load_pkl_data(args.folder_name + 'f_data_list.pkl', split=split)
    print(len(f_data))
    # e_data: [num, ext_dim]
    e_data, train_e_data, test_e_data, _ = load_mat_data(args.folder_name + 'fea2.mat', 'fea', split=split)
    # e_preprocess = MinMaxNormalization01()
    # e_preprocess.fit(train_e_data)
    # train_e_data = e_preprocess.transform(train_e_data)
    # test_e_data = e_preprocess.transform(test_e_data)
    print('preprocess train/test data...')
    #pre_process = MinMaxNormalization01_by_axis()
    if args.if_minus_mean:
        pre_process = MinMaxNormalization01_minus_mean()
        pre_process.fit(train_data)
        norm_mean_data = pre_process.transform(data)
        train_data = norm_mean_data[:split[0]]
        test_data = norm_mean_data[split[0]:]
    else:
        pre_process = MinMaxNormalization01()
        pre_process.fit(train_data)
        train_data = pre_process.transform(train_data)
        test_data = pre_process.transform(test_data)
    # embeddings
    #id_map = load_pickle(args.folder_name+'station_map.pkl')
    #num_station = len(id_map)
    num_station = data.shape[1]
    print('number of station: %d' % num_station)
    if args.pretrained_embeddings:
        print('load pretrained embeddings...')
        embeddings = get_embedding_from_file(args.folder_name+'embeddings.txt', num_station)
    else:
        print('train station embeddings via Word2Vec model...')
        trip_data = load_pickle(args.folder_name+'all_trip_data.pkl')
        word2vec_model = Word2Vec(sentences=trip_data, size=args.embedding_size)
        print('save Word2Vec model and embeddings...')
        word2vec_model.save(args.folder_name+'word2vec_model')
        word2vec_model.wv.save_word2vec_format(args.folder_name+'embeddings.txt', binary=False)
        del word2vec_model
        embeddings = get_embedding_from_file(args.folder_name+'embeddings.txt', num_station)
    train_loader = DataLoader(train_data, train_f_data, train_e_data,
                              args.input_steps, args.output_steps,
                              num_station)
    # val_loader = DataLoader(val_data, val_f_data,
    #                           args.input_steps, args.output_steps,
    #                           num_station, pre_process)
    test_loader = DataLoader(test_data, test_f_data, test_e_data,
                            args.input_steps, args.output_steps,
                            num_station)
    model = DyST3(num_station, args.input_steps, args.output_steps,
                 embedding_dim=args.embedding_size, embeddings=embeddings, ext_dim=e_data.shape[-1],
                 batch_size=args.batch_size, 
                 dynamic_context=args.dynamic_context, dynamic_spatial=args.dynamic_spatial, add_ext=args.add_ext)
    model_path = os.path.join(args.folder_name, 'model_save', args.model_save)
    solver = ModelSolver(model, train_loader, test_loader, pre_process,
                         batch_size=args.batch_size,
                         show_batches=args.show_batches,
                         n_epochs=args.n_epochs,
                         pretrained_model=args.pretrained_model_path,
                         update_rule=args.update_rule,
                         learning_rate=args.learning_rate,
                         model_path=model_path,
                         partial_pretrain=args.partial_pretrain
                         )
    results_path = os.path.join(model_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if args.pretrain:
        print '==================== begin pretrain ======================'
        w_att_1, w_att_2, w_h_in, w_h_out = solver.pretrain(os.path.join(model_path, 'pretrain_out'))
        np.save(os.path.join(model_path, 'w_att_1.npy'), w_att_1)
        np.save(os.path.join(model_path, 'w_att_2.npy'), w_att_2)
        np.save(os.path.join(model_path, 'w_h_in.npy'), w_h_in)
        np.save(os.path.join(model_path, 'w_h_out.npy'), w_h_out)
    if args.train:
        print '==================== begin training ======================'
        test_target, test_prediction = solver.train(os.path.join(model_path, 'out'))
        np.save(os.path.join(results_path, 'test_target.npy'), test_target)
        np.save(os.path.join(results_path, 'test_prediction.npy'), test_prediction)
    if args.test:
        print '==================== begin test =========================='
        test_target, test_prediction = solver.test()
        np.save(os.path.join(results_path, 'test_target.npy'), test_target)
        np.save(os.path.join(results_path, 'test_prediction.npy'), test_prediction)


if __name__ == "__main__":
    main()
