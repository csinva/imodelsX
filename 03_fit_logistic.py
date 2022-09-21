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
import data
from datasets import load_from_disk
import config
import sklearn
import warnings
from datetime import datetime


def get_dataset(checkpoint: str, ngrams: int, all_ngrams: bool, norm: bool,
                dataset, data_dir, data_dir_full, simple_tokenizer):
    """
    args.dataset_key_text: str, e.g. "sentence" for sst2
    """
    
    y_train = dataset['train']['label']
    y_val = dataset['validation']['label']
    
    # load embeddings
    if 'bert-base' in checkpoint or 'distilbert' in checkpoint or 'BERT' in checkpoint:
        if all_ngrams:
            try:
                data = pkl.load(open(oj(data_dir, 'data.pkl'), 'rb'))
            except Exception as e:
                # print("\tcouldn't find", , 'trying', data_dir_full)
                data = pkl.load(open(oj(data_dir_full, 'data.pkl'), 'rb'))

            X_train = data['X_train']
            X_val = data['X_val']
        else:
            try:
                reloaded_dataset = load_from_disk(data_dir)
            except Exception as e:
                # print("\tcouldn't find", data_dir, 'trying', data_dir_full)
                try:
                    reloaded_dataset = load_from_disk(data_dir_full)
                except Exception as e:
                    print("\tcouldn't find", data_dir, 'OR', data_dir_full)
                    print(e)
                    exit(1)
                
            X_train = np.array(reloaded_dataset['train']['embs']).squeeze()
            X_val = np.array(reloaded_dataset['validation']['embs']).squeeze()
        
            
        if args.subsample > 0:
            rng = np.random.default_rng(args.seed)
            idxs_subsample = rng.choice(X_train.shape[0], size=args.subsample, replace=False)
            X_train = X_train[idxs_subsample]
            y_train = np.array(y_train)[idxs_subsample]
        if norm:
            X_train = (X_train - data['mean']) / data['sigma']
            X_val = (X_val - data['mean']) / data['sigma']
            
        return X_train, X_val, y_train, y_val
    elif 'vectorizer' in checkpoint:
        if all_ngrams:
            lower_ngram = 1
        else:
            lower_ngram = ngrams
        if checkpoint == 'countvectorizer':
            vectorizer = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(lower_ngram, ngrams))
        elif checkpoint == 'tfidfvectorizer':
            vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer, ngram_range=(lower_ngram, ngrams))
        # vectorizer.fit(dataset['train']['sentence'])
        X_train = vectorizer.fit_transform(dataset['train'][args.dataset_key_text])
        X_val = vectorizer.transform(dataset['validation'][args.dataset_key_text])
        return X_train, X_val, y_train, y_val
    
def fit_and_score(X_train, X_val, y_train, y_val, r):
    # model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    m = LogisticRegressionCV(random_state=args.seed, refit=False, cv=cv)
    m.fit(X_train, y_train)
    r['model'] = deepcopy(m)
    
    # performance
    r['acc_train'] = m.score(X_train, y_train)
    r['acc_val'] = m.score(X_val, y_val)
    return r


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
    dir_name = data.get_dir_name(args, seed=None)
    # note, this is not in the data_dir only in the save
    # must come before adding -norm to the name!
    data_dir = oj(config.data_dir, args.dataset, dir_name)
    data_dir_full = oj(config.data_dir, args.dataset, data.get_dir_name(args, full_dset=True)) # no subsampling
    if args.norm:
        dir_name += '-norm' 
        
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
    X_train, X_val, y_train, y_val = get_dataset(args.checkpoint, args.ngrams, args.all, args.norm,
                                                 dataset, data_dir, data_dir_full, simple_tokenizer)
    if args.ngrams_test is not None:
        _, X_val, _, y_val = get_dataset(args.checkpoint, args.ngrams_test, args.all, args.norm,
            dataset, data_dir, data_dir_full, simple_tokenizer)
    r['num_features'] = X_train.shape[1]
    
    # fit and return model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_and_score(X_train, X_val, y_train, y_val, r)
    # print('r', r)
    
    
    # save
    os.makedirs(save_dir, exist_ok=True)
    pkl.dump(r, open(oj(save_dir, 'results.pkl'), 'wb'))
    print(save_dir, '\n', r, '\n-------------------SUCCESS------------------------\n\n')
    