import torch
from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import datasets
import numpy as np
import os.path
from spacy.lang.en import English
from datasets import load_from_disk
import pickle as pkl
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import analyze_helper
import dvu
dvu.set_style()
import pandas as pd
from os.path import join as oj
import string
from typing import List
import data
import matplotlib.pyplot as plt
import seaborn as sns
import config
pd.set_option('display.max_rows', None)


# set up model
def get_embs(texts: List[str], device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint) # for actually passing things to the model
    model = BertModel.from_pretrained(args.checkpoint)
    model = model.to(device)
    """Run this quickly if it fits
    tokens = tokenizer(texts, padding=args.padding, truncation=True, return_tensors="pt")    
    tokens = tokens.to(device)
    
    output = model(**tokens) # this takes a while....
    embs = output['pooler_output'].cpu().detach().numpy()
    return embs
    """

    # Slower way to run things...
    embs = []
    for i in tqdm(range(len(texts))):
        tokens = tokenizer([texts[i]], padding=args.padding, truncation=True, return_tensors="pt")
        tokens = tokens.to(device)
        output = model(**tokens) # this takes a while....
        emb = output['pooler_output'].cpu().detach().numpy()
        embs.append(emb)
    return np.array(embs).squeeze()

    

if __name__ == '__main__':
    class A:
        checkpoint = 'textattack/bert-base-uncased-SST-2'
        dataset = 'sst2'
        padding = True
    args = A()

    dataset = analyze_helper.get_sst_dataset()
    tok_simp = English().tokenizer
    tokenizer_func = lambda x: [str(x) for x in tok_simp(x)] 
    v = CountVectorizer(tokenizer=tokenizer_func)
    v.fit(dataset['train']['sentence'])
    words = sorted(list(v.vocabulary_.keys()))
    """Note that vocab is already based on words being sorted
    remap_idxs = {i: v.vocabulary_[k] for i, k in enumerate(words)}
    for i in range(len(remap_idxs)):
        assert remap_idxs[i] == i
    """

    #############################################################
    # Compute unigram embs + linear coefs
    #############################################################
    try:
        embs = pkl.load(open(oj(config.misc_dir, 'word_embs_sst_train.pkl'), 'rb'))

        df = pd.read_csv(oj(config.misc_dir, 'df_unigram_sst.csv'), index_col=0)
    except:
        embs = get_embs(words)
        os.makedirs(config.misc_dir)
        pkl.dump(embs, open(oj(config.misc_dir, 'word_embs_sst_train.pkl'), 'wb'))
        pkl.dump(words, open(oj(config.misc_dir, 'word_list_sst_train.pkl'), 'wb'))

        # countvec coefs
        matrix = v.transform(dataset['train']['sentence'])
        tot_counts = pd.DataFrame(matrix.sum(axis=0), columns=v.get_feature_names())
        m = LogisticRegressionCV()
        m.fit(matrix, dataset['train']['label'])
        coef = m.coef_.flatten() # note -- coef has not been mapped to same idxs as words

        # make df
        df = pd.DataFrame.from_dict({
            'coef': coef,
            'tot_counts': tot_counts.values.squeeze(),
            'unigram': words,
        })
        df.to_csv(oj(config.misc_dir, 'df_unigram_sst.csv'))

    #############################################################
    # **get bigram coefs and add them to a df with the corresponding unigram coefs**
    #############################################################
    try:
        df2 = pd.read_csv(oj(config.misc_dir, 'df_bigram_sst.csv'), index_col=0)
    except:
        # fit countvec model
        v2 = CountVectorizer(tokenizer=tokenizer_func, ngram_range=(2, 2))
        v2.fit(dataset['train']['sentence'])

        # countvec coefs
        matrix2 = v2.transform(dataset['train']['sentence'])
        tot_counts2 = pd.DataFrame(matrix2.sum(axis=0), columns=v2.get_feature_names())
        m2 = LogisticRegressionCV()
        m2.fit(matrix2, dataset['train']['label'])
        coef2 = m2.coef_.flatten() # note -- coef has not been mapped to same idxs as words

        df2 = pd.DataFrame.from_dict({
            'coef': coef2,
            'tot_counts': tot_counts2.values.squeeze(),
        #     'unigram': words,
            'bigram': sorted(list(v2.vocabulary_.keys()))
        })

        def find_unigram_scores(bigram):
            unigram1, unigram2 = bigram.split(' ')
            unigram1_score = df.loc[df['unigram'] == unigram1, 'coef'].iloc[0]
            unigram2_score = df.loc[df['unigram'] == unigram2, 'coef'].iloc[0]
            return unigram1, unigram2, unigram1_score, unigram2_score

        out = list(zip(*df2['bigram'].map(find_unigram_scores)))
        for (i, c) in enumerate(['unigram1', 'unigram2', 'coef1', 'coef2']):
            df2[c] = out[i]

        df2.to_csv(oj(config.misc_dir, 'df_bigram_sst.csv'))

    try:
        d = pkl.load(open(oj(config.misc_dir, 'top_interacting_words_df2.pkl'), 'rb'))
        embs2 = pkl.load(open(oj(config.misc_dir, 'embs2_sst_top_interactions.pkl'), 'rb'))
    except:
        # add some keys
        df2['interaction_score'] = abs(df2['coef'] - (df2['coef1'] + df2['coef2']))
        df2 = df2[['bigram', 'interaction_score', 'tot_counts', # reordering
                'coef', 'coef1', 'coef2', 'unigram1', 'unigram2', ]]

        # see which bigrams interact the most
        d = df2.sort_values('interaction_score', ascending=False).round(2)
        # d = d[d.tot_counts > 2]
        d = d.head(200)

        # compute embeddings for bigram
        bigrams = d['bigram'].values.tolist()
        embs2 = get_embs(bigrams)

        pkl.dump(embs2, open(oj(config.misc_dir, 'embs2_sst_top_interactions.pkl'), 'wb'))
        pkl.dump(bigrams, open(oj(config.misc_dir, 'word_list_sst_top_interactions.pkl'), 'wb'))
        pkl.dump(d, open(oj(config.misc_dir, 'top_interacting_words_df2.pkl'), 'wb'))


   ############################################################# 
   # Get trigram coefs / embs
   ############################################################# 
    try:
        df3 = pd.read_csv(oj(config.misc_dir, 'df_trigram_sst.csv'), index_col=0)
    except:
        print('computing trigram stuff...')

        # fit countvec model
        v2 = CountVectorizer(tokenizer=tokenizer_func, ngram_range=(3, 3))
        v2.fit(dataset['train']['sentence'])

        # countvec coefs
        matrix2 = v2.transform(dataset['train']['sentence'])
        tot_counts2 = pd.DataFrame(matrix2.sum(axis=0), columns=v2.get_feature_names())
        m2 = LogisticRegressionCV()
        m2.fit(matrix2, dataset['train']['label'])
        coef2 = m2.coef_.flatten() # note -- coef has not been mapped to same idxs as words
        trigrams = sorted(list(v2.vocabulary_.keys()))

        df3 = pd.DataFrame.from_dict({
            'coef': coef2,
            'tot_counts': tot_counts2.values.squeeze(),
            'trigram': trigrams
        })
        embs3 = get_embs(trigrams)

        df3.to_csv(oj(config.misc_dir, 'df_trigram_sst.csv'))
        pkl.dump(embs3, open(oj(config.misc_dir, 'embs3_sst_top_interactions.pkl'), 'wb'))
        pkl.dump(trigrams, open(oj(config.misc_dir, 'trigrams.pkl'), 'wb'))
        

print('successfully completed!')