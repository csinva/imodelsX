"""
Simple scikit-learn interface for Emb-GAM.


Emb-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models
Chandan Singh & Jianfeng Gao
https://arxiv.org/abs/2209.11799
"""
from numbers import Number
from re import I
from typing import List, Tuple
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from spacy.lang.en import English
import transformers
import embgam.embed
from functools import partial
from tqdm import tqdm
import warnings
import torch
from sklearn.exceptions import ConvergenceWarning
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmbGAM(BaseEstimator):
    """Emb-GAM Class - use either EmbGAMClassifier or EmbGAMRegressor to call these functions
    """

    def __init__(
        self,
        checkpoint: str = 'bert-base-uncased',
        layer: str = 'last_hidden_state',
        ngrams: int = 2,
        all_ngrams: bool = False,
        tokenizer_ngrams=None,
        random_state=None,
    ):
        """
        Params
        -------
        
        checkpoint
            Name of model checkpoint (i.e. to be fetch by huggingface)
        layer
            Name of layer to extract embeddings from
        ngrams
            Order of ngrams to extract. 1 for unigrams, 2 for bigrams, etc.
        all_ngrams
            Whether to use all order ngrams <= ngrams argument
        tokenizer_ngrams
            if None, defaults to spacy English tokenizer
        random_state
            random seed for fitting
        """
        self.checkpoint = checkpoint
        self.ngrams = ngrams
        if tokenizer_ngrams == None:
            self.tokenizer_ngrams = English().tokenizer
        else:
            self.tokenizer_ngrams = tokenizer_ngrams
        self.layer = layer
        self.random_state = random_state
        self.all_ngrams = all_ngrams

    def fit(self, X: ArrayLike, y: ArrayLike, verbose=True,
            cache_linear_coefs: bool = True):
        """Extract embeddings then fit linear model

        Params
        ------
        X: ArrayLike[str]
        y: ArrayLike[str]
        """

        # metadata
        if isinstance(self, ClassifierMixin):
            self.classes_ = unique_labels(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # set up model
        if verbose:
            print('initializing model...')
        model = transformers.AutoModel.from_pretrained(
            self.checkpoint).to(device)
        tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(
            self.checkpoint)

        # get embs
        if verbose:
            print('calculating embeddings...')
        embs = self.get_embs_summed(X, model, tokenizer_embeddings)

        # train linear
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if verbose:
            print('training linear model...')
        if isinstance(self, ClassifierMixin):
            self.linear = LogisticRegressionCV()
        elif isinstance(self, RegressorMixin):
            self.linear = RidgeCV()
        self.linear.fit(embs, y)

        # cache linear coefs
        if cache_linear_coefs:
            if verbose:
                print('caching linear coefs...')
            self.cache_linear_coefs(X, model, tokenizer_embeddings)

        return self

    def get_embs_summed(self, X, model, tokenizer_embeddings):
        embs = []
        for i, x in tqdm(enumerate(X)):
            emb = embgam.embed.embed_and_sum_function(
                x,
                model=model,
                ngrams=self.ngrams,
                tokenizer_embeddings=tokenizer_embeddings,
                tokenizer_ngrams=self.tokenizer_ngrams,
                checkpoint=self.checkpoint,
                layer=self.layer,
                all_ngrams=self.all_ngrams,
            )
            embs.append(emb['embs'])
        return np.array(embs).squeeze()  # num_examples x embedding_size

    def cache_linear_coefs(self, X: ArrayLike, model=None, tokenizer_embeddings=None):
        """Cache linear coefs for ngrams into a dictionary self.coefs_dict_
        If it already exists, only add linear coefs for new ngrams
        """

        if model is None:
            model = transformers.AutoModel.from_pretrained(
                self.checkpoint).to(device)
        if tokenizer_embeddings is None:
            tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(
                self.checkpoint)

        ngrams_list = self.get_ngrams_list(X)

        # dont recompute ngrams we already know
        if hasattr(self, 'coefs_dict_'):
            coefs_dict_old = self.coefs_dict_
        else:
            coefs_dict_old = {}
        ngrams_list = [ngram for ngram in ngrams_list
                       if not ngram in coefs_dict_old]
        if len(ngrams_list) == 0:
            print('\tNothing to update!')
            return

        # compute embeddings
        """
        # Faster version that needs more memory
        tokens = tokenizer(ngrams_list, padding=args.padding,
                           truncation=True, return_tensors="pt")
        tokens = tokens.to(device)

        output = model(**tokens) # this takes a while....
        embs = output['pooler_output'].cpu().detach().numpy()
        return embs
        """
        # Slower way to run things but won't run out of mem
        embs = []
        for i in tqdm(range(len(ngrams_list))):
            tokens = tokenizer_embeddings(
                [ngrams_list[i]], padding=True, truncation=True, return_tensors="pt")
            tokens = tokens.to(model.device)
            output = model(**tokens)
            emb = output[self.layer].cpu().detach().numpy()
            if len(emb.shape) == 3:  # includes seq_len
                emb = emb.mean(axis=1)
            embs.append(emb)
        embs = np.array(embs).squeeze()

        # save coefs
        coef_embs = self.linear.coef_.squeeze()
        linear_coef = embs @ coef_embs
        self.coefs_dict_ = {
            **coefs_dict_old,
            **{ngrams_list[i]: linear_coef[i]
               for i in range(len(ngrams_list))}
        }
        print('coefs_dict_ len', len(self.coefs_dict_))

    def get_ngrams_list(self, X):
        all_ngrams = set()
        for x in X:
            seqs = embgam.embed.generate_ngrams_list(
                x,
                ngrams=self.ngrams,
                tokenizer_ngrams=self.tokenizer_ngrams,
                all_ngrams=self.all_ngrams,
            )
            all_ngrams |= set(seqs)
        return sorted(list(all_ngrams))

        """
        # Approach using sklearn tokenizer (faster but not perfect match)
        def tokenizer_func(x):
            return [str(x) for x in self.tokenizer_ngrams(x)]
        v = CountVectorizer(tokenizer=tokenizer_func,
                            ngram_range=(self.ngrams, self.ngrams))
        v.fit(X)
        ngrams_list = sorted(list(v.vocabulary_.keys()))
        """

    def predict(self, X):
        '''Predict. For regression returns continuous output.
        For classification, returns discrete output.
        '''
        check_is_fitted(self)
        preds = self.predict_cached(X)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            return ((preds + self.linear.intercept_) > 0).astype(int)

    def predict_proba(self, X):
        if not isinstance(self, ClassifierMixin):
            raise Exception(
                "predict_proba only available for EmbGAMClassifier")
        check_is_fitted(self)
        preds = self.predict_cached(X)
        logits = np.vstack(
            (1 - preds, preds)).transpose()
        return softmax(logits, axis=1)

    def predict_cached(self, X):
        """Predict only the cached coefs in self.coefs_dict_
        """
        assert hasattr(self, 'coefs_dict_'), 'coefs are not cached!'
        preds = []
        n_unseen_ngrams = 0
        for x in tqdm(X):
            pred = 0
            seqs = embgam.embed.generate_ngrams_list(
                x,
                ngrams=self.ngrams,
                tokenizer_ngrams=self.tokenizer_ngrams,
                all_ngrams=self.all_ngrams,
            )
            for seq in seqs:
                if seq in self.coefs_dict_:
                    pred += self.coefs_dict_[seq]
                else:
                    n_unseen_ngrams += 1
            preds.append(pred)
        if n_unseen_ngrams > 0:
            warnings.warn(
                f'Saw an unseen ungram {n_unseen_ngrams} times. \
For better performance, call cache_linear_coefs on the test dataset \
before calling predict.')
        return np.array(preds)
    """
    def __str__(self):
        s = '> ------------------------------\n'
        s += '> EmbGAM:\n'
        s += '> \tPredictions are made by summing the coefficients of each rule\n'
        s += '> ------------------------------\n'
        return s + self.visualize().to_string(index=False) + '\n'
    """


class EmbGAMRegressor(EmbGAM, RegressorMixin):
    ...


class EmbGAMClassifier(EmbGAM, ClassifierMixin):
    ...
