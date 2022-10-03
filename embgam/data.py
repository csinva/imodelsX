import datasets
import os
from os.path import join as oj
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split


def process_data_and_args(dataset):
    """Load dataset + return the relevant dataset key
    """
    # load dset
    if dataset == 'tweet_eval':
        dset = datasets.load_dataset('tweet_eval', 'hate')
    elif dataset == 'financial_phrasebank':
        train = datasets.load_dataset('financial_phrasebank', 'sentences_75agree',
                                      revision='main', split='train')
        idxs_train, idxs_val = train_test_split(
            np.arange(len(train)), test_size=0.33, random_state=13)
        dset = datasets.DatasetDict()
        dset['train'] = train.select(idxs_train)
        dset['validation'] = train.select(idxs_val)
    else:
        dset = datasets.load_dataset(dataset)

    # process dset
    if dataset == 'sst2':
        del dset['test']
        dataset_key_text = 'sentence'
    elif dataset == 'financial_phrasebank':
        dataset_key_text = 'sentence'
    elif dataset == 'imdb':
        del dset['unsupervised']
        dset['validation'] = dset['test']
        del dset['test']
        dataset_key_text = 'text'
    elif dataset == 'emotion':
        del dset['test']
        dataset_key_text = 'text'
    elif dataset == 'rotten_tomatoes':
        del dset['test']
        dataset_key_text = 'text'
    elif dataset == 'tweet_eval':
        del dset['test']
        dataset_key_text = 'text'
    else:
        dataset_key_text = 'text'  # default
    # if args.subsample > 0:
    #    dataset['train'] = dataset['train'].select(range(args.subsample))
    return dset, dataset_key_text


def load_fitted_results(fname_filters=[], dset_filters=[],
                        drop_model=True, results_dir_main='results',):
    """filters must be included in fname to be included.
    Empty list of filters will return everything
    """
    dsets = [d for d in sorted(os.listdir(results_dir_main))
             if not d.endswith('.pkl')]
    for dset_filter in dset_filters:
        dsets = [d for d in dsets if dset_filter in d]
    rs = []
    print('dsets', dsets)
    for dset in dsets:
        print('\tprocessing', dset)
        try:
            # depending on how much is saved, this may take a while
            results_dir = oj(results_dir_main, dset)
            dir_names = sorted([fname
                                for fname in os.listdir(results_dir)
                                if os.path.isdir(oj(results_dir, fname))
                                ])

            for fname_filter in fname_filters:
                dir_names = [d for d in dir_names if fname_filter in d]
            if drop_model:
                results_list = [pd.Series(pkl.load(open(oj(results_dir, dir_name, 'results.pkl'), "rb"))).drop('model')
                                for dir_name in tqdm(dir_names)]
            else:
                results_list = [pd.Series(pkl.load(open(oj(results_dir, dir_name, 'results.pkl'), "rb")))
                                for dir_name in tqdm(dir_names)]
            # .drop(columns='model')
            r = pd.concat(results_list, axis=1).T.infer_objects()
            r['all'] = r['all'].replace('True', 'all')
            r['seed'] = r['seed'].fillna(1)
            r['layer'] = r['layer'].fillna('pooler_output')
            r = r.fillna('')
            r['dataset'] = dset
            rs.append(r)
        except Exception as e:
            print('ignoring this exception: ', e)
    rs = pd.concat(rs)
    return rs


def get_dir_name(args, full_dset=False, ngrams=None, seed=None, ngrams_test=None):
    """Get directory named for saving between embeddings / fit logistic
    """

    # handle arguments
    subsample = args.subsample
    if full_dset:
        subsample = -1
    if not ngrams:
        ngrams = args.ngrams

    # create dir_name
    dir_name = f"ngram={ngrams}_" + 'sub=' + \
        str(subsample) + '_' + args.checkpoint.replace('/', '-')  # + "_" + padding

    # append extra things
    if not args.layer == 'pooler_output':
        dir_name += '__' + args.layer
    if args.parsing:
        dir_name += '__' + args.parsing
    if seed:
        dir_name += '__' + str(seed)
    if hasattr(args, 'all') and args.all == 'all':
        dir_name += '-all'
    if ngrams_test:
        dir_name += '___ngtest=' + str(ngrams_test)
    return dir_name
