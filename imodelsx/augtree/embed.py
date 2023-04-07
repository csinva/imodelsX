import re
from typing import List

from tqdm import tqdm
import imodelsx.data
import imodelsx.util
import imodelsx.augtree.utils
import os.path
import numpy as np
from imodelsx.augtree.utils import clean_str
from os.path import join
import pickle as pkl
import sklearn.metrics
# import fire
import torch.cuda
from scipy.spatial import distance
# from numba import jit


CHECKPOINTS_DICT = {
    'financial_phrasebank': 'ahmedrachid/FinancialBERT-Sentiment-Analysis',
    'rotten_tomatoes': 'textattack/bert-base-uncased-rotten-tomatoes',
    'emotion': 'nateraw/bert-base-uncased-emotion',
    'sst2': 'textattack/bert-base-uncased-SST-2',
}

# @jit(nopython=True)
def pairwise_distances(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    dists = np.zeros((n, n))
    for i in tqdm(range(n)):
    # for i in range(n):
        vec_i = X[i]
        dists[i] = np.linalg.norm(X - vec_i, axis=1)
    return dists

class EmbsManager:
    def __init__(self,
        save_dir_embs='/home/chansingh/llm-tree/results/embs_cache',
        dataset_name: str='financial_phrasebank',
        # checkpoint: str='ahmedrachid/FinancialBERT-Sentiment-Analysis',
        ngrams: int=2,
        # metric: str='euclidean',
        n_keep: int=200,
        n_jobs: int=60,
    ):
        '''
        Params
        ------
        n_jobs
            Number of cpus for computing pairwise distances
        '''
        print(locals())
        self.save_dir_embs = save_dir_embs
        self.dataset_name = dataset_name
        self.ngrams = ngrams
        self.checkpoint = CHECKPOINTS_DICT[dataset_name]
        self.n_keep = n_keep
        self.n_jobs = n_jobs

        # cache embeddings
        dir_name_top = join(save_dir_embs, f'{clean_str(dataset_name)}___ngrams={ngrams}')
        dir_name_checkpoint = join(dir_name_top, clean_str(self.checkpoint))
        fname_vocab = join(dir_name_top, 'vocab.pkl')
        fname_mappings = join(dir_name_checkpoint, f'mappings_{"euclidean"}.npy')
        os.makedirs(dir_name_checkpoint, exist_ok=True)
        if not os.path.exists(fname_mappings):
            print(fname_mappings, 'not found')
            self._compute_mappings(fname_vocab, dir_name_checkpoint)

        # load mappings + vocab
        print('loading from cache...')
        with open(fname_mappings, 'rb') as f:
            self.mappings = np.load(f)
        with open(fname_vocab, 'rb') as f:
            self.ngrams_arr = np.array(pkl.load(f))
        return

    def _compute_mappings(self, fname_vocab: str, dir_name_checkpoint: str):
        # get raw data strings
        dset, dataset_key_text = imodelsx.data.load_huggingface_dataset(self.dataset_name, binary_classification=True)
        X = dset['train'][dataset_key_text] + dset['validation'][dataset_key_text]
        tokenizer = imodelsx.augtree.utils.get_spacy_tokenizer()

        # get ngrams list
        ngrams_list = [
            imodelsx.util.generate_ngrams_list(
                x, ngrams=self.ngrams,
                tokenizer_ngrams=tokenizer,
                all_ngrams=True
            )
            for x in X
        ]
        ngrams_list = sum(ngrams_list, [])
        ngrams_list = sorted(list(set(ngrams_list)))
        # ngrams_list = ngrams_list[:5]
        print(f'ngrams_list {len(ngrams_list)} {ngrams_list[:5]}')

        # compute embeddings
        print('computing embeddings...')
        embs = imodelsx.util.get_embs_llm(ngrams_list, self.checkpoint)
        print('embs.shape', embs.shape)
        torch.cuda.empty_cache()

        # compute embedding similarities
        with open(fname_vocab, 'wb') as f:
            pkl.dump(ngrams_list, f)
        print(f'computing embedding similarities...')
        # (N, D) -> (N, N)
        # pairwise_dists = sklearn.metrics.pairwise_distances(
            # embs, metric=metric, n_jobs=self.n_jobs)
        pairwise_dists = pairwise_distances(embs)
        pairwise_dists[np.eye(pairwise_dists.shape[0]).astype(int)] = 1e10

        # argsort each row
        # (N, N) -> (N, N)
        print(f'computing argsort...')
        args = np.zeros((len(ngrams_list), self.n_keep), dtype=int)
        for i in tqdm(range(len(ngrams_list))):
            args[i] = np.argsort(pairwise_dists[i])[:self.n_keep]
        # args = np.argsort(pairwise_dists, axis=1)[:, :self.n_keep]
        # print(pairwise_dists.sort(axis=1)[0].round(20))

        cache_file = join(dir_name_checkpoint, f'mappings_{"euclidean"}.npy')
        with open(cache_file, 'wb') as f:
            np.save(f, args)
        


    def expand_keyword(self, keyword: str, n_expands=50) -> List[str]:
        '''Expand ngram using similar keywords
        '''
        # self.ngrams_list is an array of ngrams
        # self.mappings is a numpy array of indexes into ngrams_list
        
        # find index where keyword occurs in ngrams_arr
        find_keyword = self.ngrams_arr == keyword
        # print(self.ngrams_arr[500:900])
        if find_keyword.sum() == 0:
            print(repr(keyword), 'not found')
            return []
        else:
            idx_keyword = np.argmax(find_keyword)
            try:
                idx_keyword = int(idx_keyword)
            except:
                idx_keyword = int(idx_keyword[0])
            idxs_expanded = self.mappings[idx_keyword, :n_expands]
            keywords_expanded = self.ngrams_arr[idxs_expanded]
            # print(f'{keyword=} {keywords_expanded=}')
            return keywords_expanded
        

def test_dists():
    # sample embeddings matrix
    X = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [2, 4, 6],
        [1, 2, 2],
    ])
    dists_eucl = pairwise_distances(X).round(2)
    # dists_cos = pairwise_distances(X, metric='cosine').round(2)
    assert np.min(dists_eucl) >= 0
    # assert np.min(dists_cos) >= 0
    dists_eucl_ref = sklearn.metrics.pairwise_distances(X, metric='euclidean').round(2)
    print('dists_eucl', dists_eucl)
    print('dists_eucl_ref', dists_eucl_ref)
    assert np.allclose(dists_eucl, dists_eucl_ref)
    

# if __name__ == '__main__':
    # test_dists()
    # allows calling wth args, e.g. python embed.py --dataset_name sst2
    # fire.Fire(EmbsManager)

    # embs = EmbsManager(
    #     dataset_name='financial_phrasebank',
    #     checkpoint='ahmedrachid/FinancialBERT-Sentiment-Analysis'
    # ).expand_keyword('great')

