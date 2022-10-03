from datetime import datetime
import warnings
import sklearn
import experiments.config as config
from datasets import load_from_disk
import embgam.linear
import embgam.data
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pickle as pkl
import os
from os.path import join as oj
from spacy.lang.en import English
import argparse
path_to_current_file = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':

    # hyperparams
    parser = argparse.ArgumentParser(description='Process some integers.')
    # checkpoint values: countvectorizer, tfidfvectorizer
    parser.add_argument('--checkpoint', type=str,
                        help='name of model checkpoint', default='bert-base-uncased')
    parser.add_argument('--ngrams', type=int,
                        help='dimensionality of ngrams', default=1)
    parser.add_argument('--ngrams_test', type=int,
                        help='optional, test dim of ngrams (if different from training dim of ngrams)', default=None)
    parser.add_argument('--subsample', type=int,
                        help='whether to only keep only this many training samples. \
                            Note: this is currently broken when not -1, need to make \
                                sure to shuffle samples before doing subselection', default=-1)
    parser.add_argument('--all', type=str, default='',
                        help='whether to use all-ngrams')
    parser.add_argument('--norm', type=str, default='',
                        help='whether to normalize before fitting')
    parser.add_argument('--dataset', type=str,
                        help='which dataset to fit', default='sst2')  # sst2, imdb
    parser.add_argument('--seed', type=int, help='random seed', default=1)
    parser.add_argument('--layer', type=str, help='which layer of the model to extract',
                        default='pooler_output')  # last_hidden_state_mean
    parser.add_argument('--parsing', type=str,
                        help='extra logic for parsing', default='')  # noun_chunks
    parser.add_argument('--ignore_cache', action='store_true', default=False)
    args = parser.parse_args()
    args.padding = True  # 'max_length' # True
    print('\n-------------------------------------\nfit_logistic hyperparams', vars(args))

    # check if cached (this process is needlessly complicated...)
    data_dir = oj(config.data_dir, args.dataset,
                  embgam.data.get_dir_name(args, seed=None))
    data_dir_full = oj(config.data_dir, args.dataset, embgam.data.get_dir_name(
        args, full_dset=True))  # no subsampling

    out_dir_name = embgam.data.get_dir_name(
        args, seed=args.seed, ngrams_test=args.ngrams_test)
    save_dir = oj(config.results_dir, args.dataset, out_dir_name)
    if os.path.exists(save_dir) and not args.ignore_cache:
        print('aready ran', save_dir)
        exit(0)

    # set up model
    np.random.seed(args.seed)
    tokenizer_ngrams = English().tokenizer  # for our word-finding
    if 'vectorizer' in args.checkpoint:
        def tokenizer_ngrams(x): return [str(
            x) for x in tokenizer_ngrams(x)]  # for our word-finding

    # set up data
    dataset, dataset_key_text = embgam.data.process_data_and_args(args.dataset)

    # get data
    r = vars(args)
    kwargs = dict(
        checkpoint=args.checkpoint, ngrams=args.ngrams,
        all_ngrams=args.all, norm=args.norm,
        dataset=dataset, data_dir=data_dir,
        data_dir_full=data_dir_full, tokenizer_ngrams=tokenizer_ngrams,
        seed=args.seed, subsample=-1,
        dataset_key_text=dataset_key_text,
    )
    X_train, X_val, y_train, y_val = embgam.linear.get_dataset_for_logistic(
        **kwargs)

    # test on a different dset
    if args.ngrams_test is not None:
        kwargs['data_dir'] = oj(config.data_dir, args.dataset, embgam.data.get_dir_name(
            args, seed=None, ngrams=args.ngrams_test))
        kwargs['data_dir_full'] = oj(config.data_dir, args.dataset, embgam.data.get_dir_name(
            args, ngrams=args.ngrams_test, full_dset=True))  # no subsampling
        _, X_val, _, y_val = embgam.linear.get_dataset_for_logistic(**kwargs)
    r['num_features'] = X_train.shape[1]

    # fit and return model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embgam.linear.fit_and_score_logistic(
            X_train, X_val, y_train, y_val, r, seed=args.seed)
    # print('r', r)

    # save
    os.makedirs(save_dir, exist_ok=True)
    pkl.dump(r, open(oj(save_dir, 'results.pkl'), 'wb'))
    print(save_dir, '\n', r, '\n-------------------SUCCESS------------------------\n\n')
