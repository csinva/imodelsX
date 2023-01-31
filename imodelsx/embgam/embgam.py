"""
Simple scikit-learn interface for Emb-GAM.


Emb-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models
Chandan Singh & Jianfeng Gao
https://arxiv.org/abs/2209.11799
"""
from numpy.typing import ArrayLike
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from spacy.lang.en import English
from sklearn.preprocessing import StandardScaler
import transformers
import imodelsx.embgam.embed
from tqdm import tqdm
import os
import os.path
import warnings
import pickle as pkl
import torch
from sklearn.exceptions import ConvergenceWarning
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmbGAM(BaseEstimator):
    def __init__(
        self,
        checkpoint: str = 'bert-base-uncased',
        layer: str = 'last_hidden_state',
        ngrams: int = 2,
        all_ngrams: bool = False,
        tokenizer_ngrams=None,
        random_state=None,
        normalize_embs=False,
        fit_with_ngram_decomposition=True,
    ):
        '''Emb-GAM Class - use either EmbGAMClassifier or EmbGAMRegressor rather than initializing this class directly.

        Parameters
        ----------
        checkpoint: str
            Name of model checkpoint (i.e. to be fetch by huggingface)
        layer: str
            Name of layer to extract embeddings from
        ngrams
            Order of ngrams to extract. 1 for unigrams, 2 for bigrams, etc.
        all_ngrams
            Whether to use all order ngrams <= ngrams argument
        tokenizer_ngrams
            if None, defaults to spacy English tokenizer
        random_state
            random seed for fitting
        normalize_embs
            whether to normalize embeddings before fitting linear model
        fit_with_ngram_decomposition
            whether to fit to emb-gam style (using sum of embeddings of each ngram)
            if False, fits a typical model and uses ngram decomposition only for prediction / testing
            Usually, setting this to False will considerably impede performance
        '''
        self.checkpoint = checkpoint
        self.ngrams = ngrams
        if tokenizer_ngrams == None:
            self.tokenizer_ngrams = English().tokenizer
        else:
            self.tokenizer_ngrams = tokenizer_ngrams
        self.layer = layer
        self.random_state = random_state
        self.all_ngrams = all_ngrams
        self.normalize_embs = normalize_embs
        self.fit_with_ngram_decomposition = fit_with_ngram_decomposition

    def fit(self, X: ArrayLike, y: ArrayLike, verbose=True,
            cache_linear_coefs: bool = True,
            cache_embs_dir: str = None,
            ):
        '''Extract embeddings then fit linear model

        Parameters
        ----------
        X: ArrayLike[str]
        y: ArrayLike[str]
        cache_linear_coefs
            Whether to compute and cache linear coefs into self.coefs_dict_
        cache_embs_dir, optional
            if not None, directory to save embeddings into
        '''

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
        embs = self._get_embs_summed(X, model, tokenizer_embeddings)
        if self.normalize_embs:
            self.normalizer = StandardScaler()
            embs = self.normalizer.fit_transform(embs)
        if cache_embs_dir is not None:
            os.makedirs(cache_embs_dir, exist_ok=True)
            pkl.dump(embs, open(os.path.join(cache_embs_dir, 'embs.pkl'), 'wb'))

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

    def _get_embs_summed(self, X, model, tokenizer_embeddings):
        embs = []
        for x in tqdm(X):
            emb = imodelsx.embgam.embed.embed_and_sum_function(
                x,
                model=model,
                ngrams=self.ngrams,
                tokenizer_embeddings=tokenizer_embeddings,
                tokenizer_ngrams=self.tokenizer_ngrams,
                checkpoint=self.checkpoint,
                layer=self.layer,
                all_ngrams=self.all_ngrams,
                fit_with_ngram_decomposition=self.fit_with_ngram_decomposition,
            )
            embs.append(emb['embs'])
        return np.array(embs).squeeze()  # num_examples x embedding_size

    def cache_linear_coefs(self, X: ArrayLike, model=None,
                           tokenizer_embeddings=None,
                           renormalize_embs: bool = False,
                           verbose: bool = True):
        """Cache linear coefs for ngrams into a dictionary self.coefs_dict_
        If it already exists, only add linear coefs for new ngrams

        Params
        ------
        renormalize_embs
            whether to renormalize embeddings before fitting linear model
            (useful if getting a test set that is different from the training)
        """

        if model is None:
            model = transformers.AutoModel.from_pretrained(
                self.checkpoint).to(device)
        if tokenizer_embeddings is None:
            tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(
                self.checkpoint)

        ngrams_list = self._get_ngrams_list(X)

        # dont recompute ngrams we already know
        if hasattr(self, 'coefs_dict_'):
            coefs_dict_old = self.coefs_dict_
        else:
            coefs_dict_old = {}
        ngrams_list = [ngram for ngram in ngrams_list
                       if not ngram in coefs_dict_old]
        if len(ngrams_list) == 0 and verbose:
            print('\tNothing to update!')
            return

        embs = self._get_embs(ngrams_list, model, tokenizer_embeddings)
        if renormalize_embs:
            embs = StandardScaler().fit_transform(embs)
        elif self.normalize_embs:
            embs = self.normalizer.transform(embs)

        # save coefs
        coef_embs = self.linear.coef_.squeeze().transpose()
        linear_coef = embs @ coef_embs
        self.coefs_dict_ = {
            **coefs_dict_old,
            **{ngrams_list[i]: linear_coef[i]
               for i in range(len(ngrams_list))}
        }
        if verbose:
            print('After caching, coefs_dict_ len', len(self.coefs_dict_))

    def _get_embs(self, ngrams_list, model, tokenizer_embeddings):
        """Get embeddings for a list of ngrams (not summed!)
        """
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
        return embs

        """
        # Faster version that needs more memory
        tokens = tokenizer(ngrams_list, padding=args.padding,
                           truncation=True, return_tensors="pt")
        tokens = tokens.to(device)

        output = model(**tokens) # this takes a while....
        embs = output['pooler_output'].cpu().detach().numpy()
        return embs
        """

    def _get_ngrams_list(self, X):
        all_ngrams = set()
        for x in X:
            seqs = imodelsx.util.generate_ngrams_list(
                x,
                ngrams=self.ngrams,
                tokenizer_ngrams=self.tokenizer_ngrams,
                all_ngrams=self.all_ngrams,
            )
            all_ngrams |= set(seqs)
        return sorted(list(all_ngrams))

    def predict(self, X, warn=True):
        '''For regression returns continuous output.
        For classification, returns discrete output.
        '''
        check_is_fitted(self)
        preds = self._predict_cached(X, warn=warn)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            if preds.ndim > 1:  # multiclass classification
                return np.argmax(preds, axis=1)
            else:
                return (preds + self.linear.intercept_ > 0).astype(int)

    def predict_proba(self, X, warn=True):
        if not isinstance(self, ClassifierMixin):
            raise Exception(
                "predict_proba only available for EmbGAMClassifier")
        check_is_fitted(self)
        preds = self._predict_cached(X, warn=warn)
        if preds.ndim > 1:  # multiclass classification
            logits = preds
        else:
            logits = np.vstack(
                (1 - preds, preds)).transpose()
        return softmax(logits, axis=1)

    def _predict_cached(self, X, warn):
        """Predict only the cached coefs in self.coefs_dict_
        """
        assert hasattr(self, 'coefs_dict_'), 'coefs are not cached!'
        preds = []
        n_unseen_ngrams = 0
        n_classes = len(self.classes_)
        for x in X:
            if n_classes > 2:
                pred = np.zeros(n_classes)
            else:
                pred = 0
            seqs = imodelsx.util.generate_ngrams_list(
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
        if n_unseen_ngrams > 0 and warn:
            warnings.warn(
                f'Saw an unseen ungram {n_unseen_ngrams} times. \
For better performance, call cache_linear_coefs on the test dataset \
before calling predict.')
        return np.array(preds)


class EmbGAMRegressor(EmbGAM, RegressorMixin):
    ...


class EmbGAMClassifier(EmbGAM, ClassifierMixin):
    ...
