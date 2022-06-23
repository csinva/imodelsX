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
import sklearn
import warnings


def get_dataset(checkpoint: str, ngrams: int, all_ngrams: bool, norm: bool,
                dataset, processed_dir, simple_tokenizer):
    # load embeddings
    if 'bert-base' in checkpoint or 'distilbert' in checkpoint:
        if all_ngrams:
            data = pkl.load(open(oj(processed_dir, 'data.pkl'), 'rb'))
#             if norm:
            X_train = data['X_train']
            X_val = data['X_val']
            if norm:
                X_train = (X_train - data['mean']) / data['sigma']
                X_val = (X_val - data['mean']) / data['sigma']
        else:
            reloaded_dataset = load_from_disk(processed_dir)
            X_train = np.array(reloaded_dataset['train']['embs']).squeeze()
            X_val = np.array(reloaded_dataset['validation']['embs']).squeeze()
        return X_train, X_val
    elif 'vectorizer' in checkpoint:
        if checkpoint == 'countvectorizer':
            vectorizer = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(1, ngrams))
        elif checkpoint == 'tfidfvectorizer':
            vectorizer = TfidfVectorizer(tokenizer=simple_tokenizer, ngram_range=(1, ngrams))
        # vectorizer.fit(dataset['train']['sentence'])
        X_train = vectorizer.fit_transform(dataset['train']['sentence'])
        X_val = vectorizer.transform(dataset['validation']['sentence'])
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
    args = parser.parse_args()
    args.padding = True # 'max_length' # True
    print('\n\nfit_logistic hyperparams', vars(args), '\n\n')
    
    # check if cached
    dir_name = f"ngram={args.ngrams}_" + 'sub=' + str(args.subsample) + '_' + args.checkpoint.replace('/', '-') # + "_" + padding
    if args.all == 'all':
        dir_name += '-all'
    processed_dir = oj('/scratch/users/vision/chandan/embedded-ngrams/data/processed', dir_name)
#     processed_dir = oj(path_to_current_file, 'data/processed', dir_name)
    if args.norm:
        dir_name += '-norm' # note, this is not in the processed_dir only in the save
    results_dir = '/scratch/users/vision/chandan/embedded-ngrams/results'
    save_dir = oj(results_dir, dir_name)
    if os.path.exists(save_dir):
        print('aready ran', save_dir)
        exit(0)
    
    # set up model
    nlp = English()
    simple_tokenizer = nlp.tokenizer # for our word-finding
    if 'vectorizer' in args.checkpoint:
        simple_tokenizer = lambda x: [str(x) for x in nlp.tokenizer(x)] # for our word-finding
    
    
    # set up data
    dataset = datasets.load_dataset('sst2')
    del dataset['test'] # speed things up for now
    if args.subsample > 0:
        dataset['train'] = dataset['train'].select(range(args.subsample))
        
    # get data
    r = vars(args)
    X_train, X_val = get_dataset(args.checkpoint, args.ngrams, args.all, args.norm,
                                 dataset, processed_dir, simple_tokenizer)
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
    