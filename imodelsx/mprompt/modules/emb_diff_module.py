import logging
from typing import List
import warnings
import datasets
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing
from spacy.lang.en import English
import imodelsx
import imodelsx.util
import pickle as pkl
from os.path import dirname, join
from tqdm import trange
import os.path
import torch
import re
import mprompt.methods.llm
import scipy.spatial.distance
import mprompt.data.data
from typing import Union
from langchain import PromptTemplate
from mprompt.data.data import TASKS
from InstructorEmbedding import INSTRUCTOR
modules_dir = dirname(os.path.abspath(__file__))


class EmbDiffModule():

    def __init__(
        self,
        task_str: str = 'toy_animal',
        checkpoint='gpt2-xl',
        use_instructor=True,
    ):
        """
        Params
        ------
        """
        if use_instructor:
            print(f'loading hkunlp/instructor-xl...')
            self.extract_embs = INSTRUCTOR('hkunlp/instructor-xl')
        else:
            print(f'loading {checkpoint}...')
            self.extract_embs = pipeline(
                "feature-extraction",
                model=checkpoint,
                truncation=True,
                device=0
            )
        self.use_instructor = use_instructor
        self._init_task(task_str)

    def _init_task(self, task_str: str):
        self.task_str = task_str
        if task_str in TASKS:
            task = TASKS[task_str]
            if 'target_str' in task:
                self.target_str = task['target_str']
            else:
                self.target_str = mprompt.data.data.get_groundtruth_keyword(task_str)
        else:
            warnings.warn(f'no task found for {task_str}, using {task_str} as target_str')
            self.target_str = task_str
        self.emb = self._get_emb(self.target_str)
        # embs = [
        # self._get_emb(x) for x in ['horse', 'dog', 'cat']
        # ]
        # self.emb = np.mean(embs, axis=0)

        # print('ref', self.emb.shape)

    def _get_emb(self, x: Union[str, List[str]]) -> np.ndarray:
        if self.use_instructor:
            instruction = f"Represent the short phrase for clustering: "
            if isinstance(x, str):
                embs = self.extract_embs.encode([[instruction, x]])
                return embs[0]
            elif isinstance(x, list):
                # raising this batch_size to > 32 somehow doesn't speed up the model
                embs = self.extract_embs.encode([[instruction, x_i] for x_i in x], batch_size=32)
                return embs
        else:
            if isinstance(x, list) and len(x) > 1:
                raise NotImplementedError('batching not implemented for non-instructor models')
            elif isinstance(x, str):
                x = [x]
            
            # emb is (batch_size, 1, (seq_len + 2), embedding_dim)
            # embedding_dim = 768 for bert-base-uncased and 1024 for roberta-large
            emb = np.array(self.extract_embs(x))
            return emb[0, 0].mean(axis=0)  # mean over seq_len
            # return emb[0, 0, 0] # take [CLS] token (first)

    def __call__(self, X: Union[str, List[str]], batch_size=32, verbose=True) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        if isinstance(X, str):
            X = [X]
        neg_dists = np.zeros(len(X))
        if batch_size == 1:
            for i, x in enumerate(tqdm(X)):
                emb = self._get_emb(x)
                # neg_dists[i] = - np.linalg.norm(emb - self.emb, ord=2)
                neg_dists[i] = - scipy.spatial.distance.euclidean(emb, self.emb)
        else:
            if verbose:
                loop = trange
            else:
                loop = range
            for i in loop(0, len(X), batch_size):
                batch_start = i
                batch_end = min(i + batch_size, len(X))
                batch = X[batch_start: batch_end]
                embs = self._get_emb(batch)
                neg_dists[i: i+batch_size] = - scipy.spatial.distance.cdist(embs, [self.emb], metric='euclidean').squeeze()
        return neg_dists


if __name__ == '__main__':
    mod = EmbDiffModule(
        task_str='toy_animal',
        # checkpoint='bert-base-uncased',
        checkpoint='gpt2',
    )
    # X = mod.get_relevant_data()
    X = ['horse', 'dog', 'cat']
    # X = sum([[a for a in x] for x in X], [])
    resps = mod(X[:3], batch_size=1)
    resps2 = mod(X[:3], batch_size=3)
    resps3 = mod(X[:3], batch_size=2)
    print('shapes', resps.shape, resps2.shape, resps3.shape)
    for i in range(len(X)):
        print(X[i], resps[i], resps2[i])
    assert np.allclose(resps, resps2)
    assert np.allclose(resps, resps3)
    print('X', X)
    # print(X[0][:50])
    # print(resp)
