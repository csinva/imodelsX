import numpy as np
import pickle as pkl
import os
from os.path import join as oj
from spacy.lang.en import English
import argparse
path_to_current_file = os.path.dirname(os.path.abspath(__file__))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from copy import deepcopy
import pandas as pd
import embgam.data as data
from datasets import load_from_disk
import experiments.config as config
import sklearn
import warnings
from datetime import datetime
from embgam.linear import get_dataset_for_logistic, fit_and_score_logistic


if __name__ == '__main__':
    
    # hyperparams
    parser = argparse.ArgumentParser(description='Process some integers.')
    # checkpoint values: countvectorizer, tfidfvectorizer
    parser.add_argument('--checkpoint', type=str, help='name of model checkpoint', default='bert-base-uncased')
    parser.add_argument('--ngrams', type=int, help='dimensionality of ngrams', default=1)
    parser.add_argument('--ngrams_test', type=int, help='optional, test dim of ngrams (if different from training dim of ngrams)', default=None)
    parser.add_argument('--subsample', type=int, help='whether to only keep only this many training samples', default=-1)
    parser.add_argument('--all', type=str, default='', help='whether to use all-ngrams')
    parser.add_argument('--norm', type=str, default='', help='whether to normalize before fitting')
    parser.add_argument('--dataset', type=str, help='which dataset to fit', default='sst2') # sst2, imdb
    parser.add_argument('--seed', type=int, help='random seed', default=1)
    parser.add_argument('--layer', type=str, help='which layer of the model to extract', default='pooler_output') # last_hidden_state_mean
    parser.add_argument('--parsing', type=str, help='extra logic for parsing', default='') # noun_chunks
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    args = parser.parse_args()
    args.padding = True # 'max_length' # True
    print('\n-------------------------------------\nfit_logistic hyperparams', vars(args))
    
    # check if cached
    data_dir = oj(config.data_dir, args.dataset, data.get_dir_name(args, seed=None))
    data_dir_full = oj(config.data_dir, args.dataset, data.get_dir_name(args, full_dset=True)) # no subsampling
        
    out_dir_name = data.get_dir_name(args, seed=args.seed, ngrams_test=args.ngrams_test)
    save_dir = oj(config.results_dir, args.dataset, out_dir_name)
    if os.path.exists(save_dir) and not args.ignore_cache:
        print('aready ran', save_dir)
        exit(0)
    
    # set up model
    np.random.seed(args.seed)
    nlp = English()
    simple_tokenizer = nlp.tokenizer # for our word-finding
    if 'vectorizer' in args.checkpoint:
        simple_tokenizer = lambda x: [str(x) for x in nlp.tokenizer(x)] # for our word-finding
    
    
    # set up data
    dataset, args = data.process_data_and_args(args)
        
    # get data
    r = vars(args)
    X_train, X_val, y_train, y_val = get_dataset_for_logistic(args.checkpoint, args.ngrams, args.all, args.norm,
                                                 dataset, data_dir, data_dir_full, simple_tokenizer)
    if args.ngrams_test is not None:
        data_dir_ng = oj(config.data_dir, args.dataset, data.get_dir_name(args, seed=None, ngrams=args.ngrams_test)) 
        data_dir_full_ng = oj(config.data_dir, args.dataset, data.get_dir_name(args, ngrams=args.ngrams_test, full_dset=True)) # no subsampling
        _, X_val, _, y_val = get_dataset_for_logistic(args.checkpoint, args.ngrams_test, args.all, args.norm,
            dataset, data_dir_ng, data_dir_full_ng, simple_tokenizer)
    r['num_features'] = X_train.shape[1]
    
    # fit and return model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_and_score_logistic(X_train, X_val, y_train, y_val, r)
    # print('r', r)
    
    
    # save
    os.makedirs(save_dir, exist_ok=True)
    pkl.dump(r, open(oj(save_dir, 'results.pkl'), 'wb'))
    print(save_dir, '\n', r, '\n-------------------SUCCESS------------------------\n\n')
    