"""
Simple scikit-learn interface for Aug-Linear.

Augmenting Interpretable Models with LLMs during Training
Chandan Singh, Armin Askari, Rich Caruana, Jianfeng Gao
https://arxiv.org/abs/2209.11799
"""
from numpy.typing import ArrayLike
import numpy as np
import numpy.linalg
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeCV, Ridge
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import transformers
import imodelsx.auglinear.embed
from tqdm import tqdm
import os
from copy import deepcopy
from typing import Dict
import os.path
from typing import List
import warnings
import pickle as pkl
from os.path import join
import torch
from sklearn.exceptions import ConvergenceWarning
from imodelsx.auglinear.embed import _clean_np_array

device = "cuda" if torch.cuda.is_available() else "cpu"


class AugLinear(BaseEstimator):
    def __init__(
        self,
        checkpoint: str = "bert-base-uncased",
        layer: str = "last_hidden_state",
        ngrams: int = 2,
        all_ngrams: bool = False,
        min_frequency: int = 1,
        tokenizer_ngrams=None,
        random_state=None,
        normalize_embs=False,
        cache_embs_dir: str = None,
        fit_with_ngram_decomposition=True,
        embedding_prefix="Represent the short phrase for sentiment classification: ",
        embedding_suffix="",
        embedding_ngram_strategy='mean',
        zeroshot_class_dict: Dict[int, str] = None,
        zeroshot_strategy: str = 'pos_class',
        prune_stopwords: bool = False,
    ):
        """AugLinear Class - use either AugLinearClassifier or AugLinearRegressor rather than initializing this class directly.

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
        min_frequency
            minimum frequency of ngrams to be kept in the ngrams list.
        tokenizer_ngrams
            if None, defaults to spacy English tokenizer
        random_state
            random seed for fitting
        normalize_embs
            whether to normalize embeddings before fitting linear model
        cache_embs_dir: str = None,
            if not None, directory to save embeddings into
        fit_with_ngram_decomposition
            whether to fit to aug-linear style (using sum of embeddings of each ngram)
            if False, fits a typical model and uses ngram decomposition only for prediction / testing
            Usually, setting this to False will considerably impede performance
        embedding_prefix
            if checkpoint is an instructor/autoregressive model, prepend this prompt
        embedding_suffix
            if checkpoint is an autoregressive model, append this prompt
        embedding_ngram_strategy
            'mean': compute mean over ngram tokens
            'next_token_distr': use next token distribution as an embedding (requires AutoModelForCausalLM checkpoint)
        zeroshot_class_dict
            Maps class numbers to names of the class to use to compute the embedding
            Ex. {0: 'negative', 1: 'positive'}
        zeroshot_strategy
            'pos_class' or 'difference'
        prune_stopwords
            Whether to prune stopwords and ngrams with length < 3
        """
        self.checkpoint = checkpoint
        self.ngrams = ngrams
        if tokenizer_ngrams == None:
            from spacy.lang.en import English
            self.tokenizer_ngrams = English().tokenizer
        else:
            self.tokenizer_ngrams = tokenizer_ngrams
        self.layer = layer
        self.random_state = random_state
        self.all_ngrams = all_ngrams
        self.min_frequency = min_frequency
        self.normalize_embs = normalize_embs
        self.cache_embs_dir = cache_embs_dir
        self.fit_with_ngram_decomposition = fit_with_ngram_decomposition
        self.embedding_prefix = embedding_prefix
        self.embedding_suffix = embedding_suffix
        self.embedding_ngram_strategy = embedding_ngram_strategy
        self.zeroshot_class_dict = zeroshot_class_dict
        self.zeroshot_strategy = zeroshot_strategy
        self.prune_stopwords = prune_stopwords

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        verbose=True,
        cache_linear_coefs: bool = True,
        batch_size: int = 8,
    ):
        """Extract embeddings then fit linear model

        Parameters
        ----------
        X: ArrayLike[str]
        y: ArrayLike[str]
        cache_linear_coefs
            Whether to compute and cache linear coefs into self.coefs_dict_
        batch_size, optional
            if not None, batch size to pass while calculating embeddings
        """

        # metadata
        if isinstance(self, ClassifierMixin):
            self.classes_ = unique_labels(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # set up model
        if verbose:
            print("initializing model...")
        model, tokenizer_embeddings = self._get_model_and_tokenizer()

        # if zero-shot, then set linear and return
        if self.zeroshot_class_dict is not None:
            self._fit_zeroshot(model, tokenizer_embeddings, verbose=verbose)
            return self

        # get embs
        if verbose:
            print("calculating embeddings...")
        if self.cache_embs_dir is not None and os.path.exists(
            os.path.join(self.cache_embs_dir, "embs_train.pkl")
        ):
            embs = pkl.load(
                open(os.path.join(self.cache_embs_dir, "embs_train.pkl"), "rb")
            )
        else:
            embs = self._get_embs(
                X, model, tokenizer_embeddings, batch_size, summed=True)
            if self.cache_embs_dir is not None:
                os.makedirs(self.cache_embs_dir, exist_ok=True)
                pkl.dump(
                    embs,
                    open(os.path.join(self.cache_embs_dir, "embs_train.pkl"), "wb"),
                )

        # normalize embs
        if self.normalize_embs:
            self.normalizer = StandardScaler()
            embs = self.normalizer.fit_transform(embs)

        # train linear
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if verbose:
            print("set up linear model...")
        if isinstance(self, ClassifierMixin):
            self.linear = LogisticRegressionCV()
        elif isinstance(self, RegressorMixin):
            self.linear = RidgeCV()
        self.linear.fit(embs, y)

        # cache linear coefs
        if cache_linear_coefs:
            if verbose:
                print("caching linear coefs...")
            self.cache_linear_coefs(X, model, tokenizer_embeddings)

        return self

    def _get_model_and_tokenizer(self):
        if self.checkpoint.startswith("hkunlp/instructor-xl"):
            from InstructorEmbedding import INSTRUCTOR
            model = INSTRUCTOR(self.checkpoint)
            tokenizer_embeddings = None
        else:
            tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(
                self.checkpoint
            )
            if self.embedding_ngram_strategy == 'next_token_distr':
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            else:
                model = transformers.AutoModel.from_pretrained(
                    self.checkpoint).to(device)

        return model.eval(), tokenizer_embeddings

    def cache_linear_coefs(
        self,
        X: ArrayLike,
        model=None,
        tokenizer_embeddings=None,
        renormalize_embs_strategy: str = None,
        batch_size: int = 8,
        verbose: bool = True,
        batch_size_embs: int = 512,
    ):
        """Cache linear coefs for ngrams into a dictionary self.coefs_dict_
        If it already exists, only add linear coefs for new ngrams

        Params
        ------
        renormalize_embs_strategy
            whether to renormalize embeddings before fitting linear model
            (useful if getting a test set that is different from the training)
            values: 'StandardScaler', 'QuantileTransformer'
        batch_size
            batch size to use for calculating embeddings (on gpu at same time)
        batch_size_embs
            batch size to use for number of embeddings stored (on cpu at same time)
        """
        assert renormalize_embs_strategy in [
            None, "StandardScaler", "QuantileTransformer", 'None']
        model, tokenizer_embeddings = self._get_model_and_tokenizer()

        ngrams_list = self._get_unique_ngrams_list(X)

        # dont recompute ngrams we already know
        if hasattr(self, "coefs_dict_"):
            coefs_dict_old = self.coefs_dict_
        else:
            coefs_dict_old = {}
        ngrams_list = [
            ngram for ngram in ngrams_list if not ngram in coefs_dict_old]
        if len(ngrams_list) == 0 and verbose:
            print("\tNothing to update!")
            return

        def normalize_embs(embs, renormalize_embs_strategy):
            if renormalize_embs_strategy in ["StandardScaler", "QuantileTransformer"]:
                if renormalize_embs_strategy == "StandardScaler":
                    embs = StandardScaler().fit_transform(embs)
                elif renormalize_embs_strategy == "QuantileTransformer":
                    embs = QuantileTransformer().fit_transform(embs)
            elif self.normalize_embs:
                embs = self.normalizer.transform(embs)
            return _clean_np_array(embs)

        # calculate linear coefs for each ngram in ngrams_list
        if batch_size_embs is not None:
            coef_embs = self.linear.coef_.squeeze().transpose()
            n_outputs = 1 if coef_embs.ndim == 1 else coef_embs.shape[1]
            linear_coef = np.zeros(shape=(len(ngrams_list), n_outputs))
            # calculate linear coefs in batches
            for i in tqdm(range(0, len(ngrams_list), batch_size_embs)):
                embs = self._get_embs(
                    ngrams_list[i: i + batch_size_embs],
                    model,
                    tokenizer_embeddings,
                    batch_size,
                    summed=False
                )
                embs = normalize_embs(embs, renormalize_embs_strategy)
                linear_coef[i: i + batch_size_embs] = (embs @ coef_embs).reshape(
                    -1, n_outputs
                )
        else:
            embs = self._get_embs(ngrams_list, model,
                                  tokenizer_embeddings, batch_size, summed=False)
            embs = normalize_embs(embs, renormalize_embs_strategy)
            linear_coef = embs @ coef_embs

        # save coefs
        linear_coef = linear_coef.squeeze()
        self.coefs_dict_ = {
            **coefs_dict_old,
            **{ngrams_list[i]: linear_coef[i] for i in range(len(ngrams_list))},
        }
        if verbose:
            print(
                f"\tAfter caching, len(coefs_dict_)={len(self.coefs_dict_)}, up from {len(coefs_dict_old)}")

    def _get_embs(self, X: List[str], model, tokenizer_embeddings, batch_size=8, summed=True):
        '''
        Returns
        -------
        embs: np.array
            num_examples x embedding_size
        '''
        kwargs = dict(
            model=model, tokenizer_embeddings=tokenizer_embeddings, tokenizer_ngrams=self.tokenizer_ngrams,
            checkpoint=self.checkpoint, layer=self.layer, batch_size=batch_size,
            embedding_prefix=self.embedding_prefix, embedding_suffix=self.embedding_suffix,
            prune_stopwords=self.prune_stopwords,
            embedding_strategy=self.embedding_ngram_strategy
        )

        if summed:
            embs = []
            for x in tqdm(X):
                emb = imodelsx.auglinear.embed.embed_and_sum_function(
                    x,
                    ngrams=self.ngrams,
                    all_ngrams=self.all_ngrams,
                    fit_with_ngram_decomposition=self.fit_with_ngram_decomposition,
                    **kwargs,
                )
                embs.append(emb["embs"])
            return _clean_np_array(np.array(embs).squeeze())
        else:
            # get embedding for a list of ngrams
            embs = imodelsx.auglinear.embed.embed_and_sum_function(
                X, ngrams=None, fit_with_ngram_decomposition=False, sum_embeddings=False, **kwargs,
            )["embs"]
            embs = np.array(embs).squeeze()
            assert embs.shape[0] == len(X)
            return _clean_np_array(embs)

    def _get_unique_ngrams_list(self, X):
        all_ngrams = set()
        for x in X:
            seqs = imodelsx.util.generate_ngrams_list(
                x,
                ngrams=self.ngrams,
                tokenizer_ngrams=self.tokenizer_ngrams,
                all_ngrams=self.all_ngrams,
                min_frequency=self.min_frequency,
                prune_stopwords=self.prune_stopwords,
            )
            all_ngrams |= set(seqs)
        return sorted(list(all_ngrams))

    def predict(self, X, warn=True):
        """For regression returns continuous output.
        For classification, returns discrete output.
        """

        check_is_fitted(self)
        preds = self._predict_cached(X, warn=warn)
        if isinstance(self, RegressorMixin):
            return preds
        elif isinstance(self, ClassifierMixin):
            # multiclass classification
            if preds.ndim > 1:
                return np.argmax(preds, axis=1)
            else:
                return (preds + self.linear.intercept_ > 0).astype(int)

    def predict_proba(self, X, warn=True):
        if not isinstance(self, ClassifierMixin):
            raise Exception("predict_proba only available for Classifier")
        check_is_fitted(self)
        preds = self._predict_cached(X, warn=warn)
        if preds.ndim == 1 or preds.shape[1] == 1:
            logits = np.vstack(
                (1 - preds.squeeze(), preds.squeeze())).transpose()
        else:  # multiclass classification
            logits = preds
        return softmax(logits, axis=1)

    def _predict_cached(self, X, warn=False):
        """Predict only the cached coefs in self.coefs_dict_"""
        assert hasattr(self, "coefs_dict_"), "coefs are not cached!"
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
                prune_stopwords=self.prune_stopwords,
            )
            for seq in seqs:
                if seq in self.coefs_dict_:
                    pred += self.coefs_dict_[seq]
                else:
                    n_unseen_ngrams += 1
            preds.append(pred)
        if n_unseen_ngrams > 0 and warn:
            warnings.warn(
                f"Saw an unseen ungram {n_unseen_ngrams} times. \
For better performance, call cache_linear_coefs on the test dataset \
before calling predict."
            )
        return np.array(preds).squeeze()

    def _fit_zeroshot(self, model, tokenizer_embeddings, verbose):
        if verbose:
            print("setting up zero-shot linear model...")
        if len(self.zeroshot_class_dict) > 2:
            raise NotImplementedError(
                'Only binary classification supported for zero-shot')
        embs_dict = {}
        for i, class_num in enumerate(self.zeroshot_class_dict):
            class_names = self.zeroshot_class_dict[class_num]
            if not isinstance(class_names, list):
                class_names = [class_names]
            embs_class = (
                self._get_embs(
                    class_names,
                    model, tokenizer_embeddings,
                    summed=False,
                )
                .reshape((len(class_names), -1))
                .mean(axis=0).squeeze()
            )
            embs_dict[i] = deepcopy(embs_class)

        # take pos class or take difference?
        if self.zeroshot_strategy == 'pos_class':
            emb = embs_dict[1].squeeze()
        elif self.zeroshot_strategy == 'difference':
            emb = (embs_dict[1] - embs_dict[0]).squeeze()

        # set up linear model
        if isinstance(self, ClassifierMixin):
            self.linear = LogisticRegression()
        elif isinstance(self, RegressorMixin):
            self.linear = Ridge()
        self.linear.coef_ = emb / np.linalg.norm(emb)  # - embs[0]
        # self.linear.coef_ -= np.mean(self.linear.coef_)
        # self.linear.coef_ /= np.max(np.abs(self.linear.coef_))
        self.linear.intercept_ = 0  # -np.mean(np.abs(self.linear.coef_))
        return self


class AugLinearRegressor(AugLinear, RegressorMixin):
    ...


class AugLinearClassifier(AugLinear, ClassifierMixin):
    ...
