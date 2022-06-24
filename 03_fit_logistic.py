import datasets
import numpy as np
import pickle as pkl
import os
from os.path import join as oj
from spacy.lang.en import English
import argparse
path_to_current_file = os.path.dirname(os.path.abspath(__file__))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
from copy import deepcopy
import pandas as pd
from datasets import load_from_disk
import config
import sklearn
import warnings


def get_dataset(checkpoint: str, ngrams: int, all_ngrams: bool, norm: bool,
                dataset, data_dir, data_dir_full, simple_tokenizer):
    """
    args.dataset_key_text: str, e.g. "sentence" for sst2
    """
    
    # load embeddings
    if 'bert-base' in checkpoint or 'distilbert' in checkpoint:
        if all_ngrams:
            try:
                data = pkl.load(open(oj(data_dir, 'data.pkl'), 'rb'))
            except Exception as e:
                print("\tcouldn't find", data_dir, 'trying', data_dir_full)
                data = pkl.load(open(oj(data_dir_full, 'data.pkl'), 'rb'))

            X_train = data['X_train']
            X_val = data['X_val']
        else:
            try:
                reloaded_dataset = load_from_disk(data_dir)
            except Exception as e:
                print("\tcouldn't find", data_dir, 'trying', data_dir_full)
                reloaded_dataset = load_from_disk(data_dir_full)
            X_train = np.array(reloaded_dataset['train']['embs']).squeeze()
            X_val = np.array(reloaded_dataset['validation']['embs']).squeeze()
            
        if args.subsample > 0:
            X_train = X_train[:args.subsample]

        if norm:
            X_train = (X_train - data['mean']) / data['sigma']
            X_val = (X_val - data['mean']) / data['sigma']
            
        return X_train, X_val
    elif 'vectorizer' in checkpoint:
        if checkpoint == 'countvectorizer':
            vectorizer = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(1, ngrams))
        elif checkpoint == 'tfidfvectorizer':
            vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer, ngram_range=(1, ngrams))
        # vectorizer.fit(dataset['train']['sentence'])
        X_train = vectorizer.fit_transform(dataset['train'][args.dataset_key_text])
        X_val = vectorizer.transform(dataset['validation'][args.dataset_key_text])
        return X_train, X_val
    
def fit_and_score(X_train, X_val, dataset, r):
    # model
    m = LogisticRegressionCV()
    m.fit(X_train, dataset['train']['label'])
    r['model'] = deepcopy(m)
    
    # performance
    r['acc_train'] = m.score(X_train, dataset['train']['label'])
    r['acc_val'] = m.score(X_val, dataset['validation']['label'])
    return r

def get_dir_name(args, full_dset=False):
    subsample = args.subsample
    if full_dset:
        subsample = -1
    dir_name = f"ngram={args.ngrams}_" + 'sub=' + str(subsample) + '_' + args.checkpoint.replace('/', '-') # + "_" + padding
    if args.all == 'all':
        dir_name += '-all'
    return dir_fname

if __name__ == '__main__':
    
    # hyperparams
    # models
    # "bert-base-uncased", 'textattack/bert-base-uncased-SST-2'
    # distilbert-base-uncased, , "distilbert-base-uncased-finetuned-sst-2-english"
    parser = argparse.ArgumentParser(description='Process some integers.')
    # checkpoint values: countvectorizer, tfidfvectorizer
    parser.add_argument('--checkpoint', type=str, help='name of model checkpoint', default='bert-base-uncased')
    parser.add_argument('--ngrams', type=int, help='dimensionality of ngrams', default=1)
    parser.add_argument('--subsample', type=int, help='whether to only keep only this many training samples', default=-1)
    parser.add_argument('--all', type=str, default='', help='whether to use all-ngrams')
    parser.add_argument('--norm', type=str, default='', help='whether to normalize before fitting')
    parser.add_argument('--dataset', type=str, help='which dataset to fit', default='sst2') # sst2, imdb
    args = parser.parse_args()
    args.padding = True # 'max_length' # True
    print('\n\nfit_logistic hyperparams', vars(args), '\n\n')
    
    # check if cached
    dir_name = get_dir_name(args)
    if args.all == 'all':
        dir_name += '-all'
    # note, this is not in the data_dir only in the save
    # must come before adding -norm to the name!
    data_dir = oj(config.data_dir, args.dataset, dir_name)
    data_dir_full = get_dir_name(args, full_dset=True) # no subsampling
#     if args.norm:
#         dir_name += '-norm' 
    save_dir = oj(config.results_dir, args.dataset, dir_name)
    if os.path.exists(save_dir):
        print('aready ran', save_dir)
        exit(0)
    
    # set up model
    nlp = English()
    simple_tokenizer = nlp.tokenizer # for our word-finding
    if 'vectorizer' in args.checkpoint:
        simple_tokenizer = lambda x: [str(x) for x in nlp.tokenizer(x)] # for our word-finding
    
    
    # set up data
    dataset = datasets.load_dataset(args.dataset)
    if args.dataset == 'sst2':
        del dataset['test'] # speed things up for now
        args.dataset_key_text = 'sentence'
    elif args.dataset == 'imdb':
        del dataset['unsupervised'] # speed things up for now
        dataset['validation'] = dataset['test']
        del dataset['test']
        args.dataset_key_text = 'text'
    if args.subsample > 0:
        dataset['train'] = dataset['train'].select(range(args.subsample))
        
    # get data
    r = vars(args)
    X_train, X_val = get_dataset(args.checkpoint, args.ngrams, args.all, args.norm,
                                 dataset, data_dir, data_dir_full, simple_tokenizer)
    r['num_features'] = X_train.shape[1]
    
    # fit and return model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_and_score(X_train, X_val, dataset, r)
    print('r', r)
    
    
    # save
    os.makedirs(save_dir, exist_ok=True)
    pkl.dump(r, open(oj(save_dir, 'results.pkl'), 'wb'))
    print('success', r, '\n-------------------------------------', '\n\n')
    