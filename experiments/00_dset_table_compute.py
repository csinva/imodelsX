import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
import embgam.data as data
import pickle as pkl
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

if __name__ == '__main__':
    tok_simp = English().tokenizer # init here to speedup call
    simple_tokenizer = lambda x: [str(x) for x in tok_simp(x)] 
    ds = defaultdict(list)
    r = defaultdict(list)
    class Args:
        ...

    args = Args()
    args.dataset = ''
    ks = sorted(['emotion', 'financial_phrasebank', 'rotten_tomatoes', 'sst2', 'tweet_eval'])
    for k in ks:
        args.dataset = k
        d, args = data.process_data_and_args(args)
        text = d['train'][args.dataset_key_text]
        ds['n_train'].append(len(text))


        counts = np.unique(d['train']['label'], return_counts=True)[1]
        ds['imbalance'].append(max(counts) / sum(counts))

        ds['num_classes'].append(counts.size)

        text_val = d['validation'][args.dataset_key_text]
        ds['n_val'].append(len(text_val))    

        v = CountVectorizer(tokenizer=simple_tokenizer)
        v.fit(text)
        ds['n_tokens'].append(len(v.vocabulary_))
        # r[f'{k}_unigram'].append(deepcopy(v.vocabulary_))

        v = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(2, 2))
        v.fit(text)
        ds['n_bigrams'].append(len(v.vocabulary_))
        # r[f'{k}_bigram'].append(deepcopy(v.vocabulary_))

        v = CountVectorizer(tokenizer=simple_tokenizer, ngram_range=(3, 3))
        matrix = v.fit_transform(text)
        ds['n_trigrams'].append(len(v.vocabulary_))    
        r[f'{k}_trigram'].append(matrix.sum(axis=0))    
        
        

    df = pd.DataFrame.from_dict(ds)
    df.index = ks
    df
    df.to_csv('results/datasets_ovw.csv')

    pkl.dump(r, open('results/datasets_ovw.pkl', 'wb'))