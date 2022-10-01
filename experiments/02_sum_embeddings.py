"""
We're going to aggregate embeddings run for a particular order of n-gram here (we'll just add them up).
"""

from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import numpy as np
import pickle as pkl
import os
from os.path import join as oj
from spacy.lang.en import English
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from datasets import load_from_disk
import sklearn
import warnings
import experiments.config as config


if __name__ == '__main__':
    for dset in sorted(os.listdir(config.data_dir))[::-1]:
        processed_dir = oj(config.data_dir, dset)
        dir_names = [f for f in sorted(os.listdir(processed_dir))
                    if not '-all' in f
                    ]

        print('----------Dset', dset, 'Starting---------------')
        for i in tqdm(range(len(dir_names))):
            s = dir_names[i]
            start = s.index('=') + 1
            end = s.index('_')
            num = int(s[start: end])
    #         if num > 1:
            print('\tTrying', s)
            s_new = s + '-all'
            if os.path.exists(oj(processed_dir, s_new)):
                print('\t\tdone already!')
                continue

            pre = s[:start]
            end = s[end:]
            all_exist = True
            num_missing = []
            for num_small in range(1, num):
                fname_small = pre + str(num_small) + end
                all_exist = all_exist and (fname_small in dir_names)
                if not fname_small in dir_names:
                    num_missing.append(num_small)

            if not all_exist:
                print('\t\tmissing small_ngrams: ', num_missing)
                continue

            # load dset
            reloaded_dataset = load_from_disk(oj(processed_dir, s))
            X_train = np.array(reloaded_dataset['train']['embs']).squeeze()
            X_val = np.array(reloaded_dataset['validation']['embs']).squeeze()

            for num_small in range(1, num + 1):
                fname_small = pre + str(num_small) + end            
                reloaded_dataset = load_from_disk(oj(processed_dir, fname_small))
                X_train_small = np.array(reloaded_dataset['train']['embs']).squeeze()
                X_val_small = np.array(reloaded_dataset['validation']['embs']).squeeze()

                X_train += X_train_small
                X_val += X_val_small

            os.makedirs(oj(processed_dir, s_new), exist_ok=True)
            mu = X_train.mean(axis=0)
            sigma = X_train.std(axis=0)
            r = {
                'X_train': X_train,
                'X_val': X_val,
                'mean': mu,
                'sigma': sigma,
            }
            pkl.dump(r, open(oj(processed_dir, s_new, 'data.pkl'), 'wb'))
            print('\t\tsuccess!')
        print('----------Dset', dset, 'Finished---------------')

    print('ALL DONE!')