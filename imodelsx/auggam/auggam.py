"""
Simple scikit-learn interface for Emb-GAM.


Aug-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models
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
import imodelsx.auggam.embed
from tqdm import tqdm
import os
import os.path
import warnings
import pickle as pkl
from os.path import join
import torch
from transformers import LlamaModel, LlamaTokenizer
from sklearn.exceptions import ConvergenceWarning

device = "cuda" if torch.cuda.is_available() else "cpu"


class AugGAM(BaseEstimator):
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
        cache_coefs_dir: str = None,
        fit_with_ngram_decomposition=True,
        instructor_prompt=None,
    ):
        """AugGAM-GAM Class - use either AugGAMClassifier or AugGAMRegressor rather than initializing this class directly.

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
            whether to fit to emb-gam style (using sum of embeddings of each ngram)
            if False, fits a typical model and uses ngram decomposition only for prediction / testing
            Usually, setting this to False will considerably impede performance
        instructor_prompt
            if not None, use instructor-xl with this prompt
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
        self.min_frequency = min_frequency
        self.normalize_embs = normalize_embs
        self.cache_embs_dir = cache_embs_dir
        self.fit_with_ngram_decomposition = fit_with_ngram_decomposition
        self.instructor_prompt = instructor_prompt

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
            embs = self._get_embs_summed(
                X, model, tokenizer_embeddings, batch_size)
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
            print("training linear model...")
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

    def _get_embs_summed(self, X, model, tokenizer_embeddings, batch_size):
        embs = []
        for x in tqdm(X):
            emb = imodelsx.auggam.embed.embed_and_sum_function(
                x,
                model=model,
                ngrams=self.ngrams,
                tokenizer_embeddings=tokenizer_embeddings,
                tokenizer_ngrams=self.tokenizer_ngrams,
                checkpoint=self.checkpoint,
                layer=self.layer,
                all_ngrams=self.all_ngrams,
                fit_with_ngram_decomposition=self.fit_with_ngram_decomposition,
                instructor_prompt=self.instructor_prompt,
                batch_size=batch_size,
            )
            embs.append(emb["embs"])
        return np.array(embs).squeeze()  # num_examples x embedding_size

    def _get_model_and_tokenizer(self):
        if self.checkpoint.startswith("hkunlp/instructor-xl"):
            from InstructorEmbedding import INSTRUCTOR

            model = INSTRUCTOR(self.checkpoint).to(device)
            tokenizer_embeddings = None
        elif "llama" in self.checkpoint:
            # path to extracted llama weights
            LLAMA_DIR = join(os.path.expanduser("~"), "llama")
            tokenizer_embeddings = transformers.LlamaTokenizer.from_pretrained(
                join(LLAMA_DIR, self.checkpoint)
            )
            model = transformers.LlamaModel.from_pretrained(
                join(LLAMA_DIR, self.checkpoint),
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            model = transformers.AutoModel.from_pretrained(
                self.checkpoint).to(device)
            tokenizer_embeddings = transformers.AutoTokenizer.from_pretrained(
                self.checkpoint
            )
        return model, tokenizer_embeddings

    def cache_linear_coefs(
        self,
        X: ArrayLike,
        model=None,
        tokenizer_embeddings=None,
        renormalize_embs: bool = False,
        batch_size: int = 8,
        verbose: bool = True,
        batch_size_embs: int = 512,
    ):
        """Cache linear coefs for ngrams into a dictionary self.coefs_dict_
        If it already exists, only add linear coefs for new ngrams

        Params
        ------
        renormalize_embs
            whether to renormalize embeddings before fitting linear model
            (useful if getting a test set that is different from the training)
        batch_size
            batch size to use for calculating embeddings (on gpu at same time)
        batch_size_embs
            batch size to use for number of embeddings stored (on cpu at same time)
        """
        model, tokenizer_embeddings = self._get_model_and_tokenizer()

        ngrams_list = self._get_ngrams_list(X)

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

        def normalize_embs(embs, renormalize_embs):
            if renormalize_embs:
                embs = StandardScaler().fit_transform(embs)
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
                )
                embs = normalize_embs(embs, renormalize_embs)
                linear_coef[i: i + batch_size_embs] = (embs @ coef_embs).reshape(
                    -1, n_outputs
                )
        else:
            embs = self._get_embs(ngrams_list, model,
                                  tokenizer_embeddings, batch_size)
            embs = normalize_embs(embs, renormalize_embs)
            linear_coef = embs @ coef_embs

        # save coefs
        self.coefs_dict_ = {
            **coefs_dict_old,
            **{ngrams_list[i]: linear_coef[i] for i in range(len(ngrams_list))},
        }
        if verbose:
            print("\tAfter caching, coefs_dict_ len", len(self.coefs_dict_))

    def _get_embs(self, ngrams_list, model, tokenizer_embeddings, batch_size):
        """Get embeddings for a list of ngrams (not summed!)"""
        embs = []
        if self.checkpoint.startswith("hkunlp/instructor-xl"):
            # INSTRUCTION = "Represent the short phrase for sentiment classification: "
            # embs = model.encode([[INSTRUCTION, x_i] for x_i in ngrams_list], batch_size=32)
            embs = []
            batch_size = 32
            for i in tqdm(range(0, len(ngrams_list), batch_size)):
                # ngram = ngrams_list[i]
                # embs.append(model.encode([[INSTRUCTION, ngram]])[0])
                ngram_batch = ngrams_list[i: i + batch_size]
                embs_batch = model.encode(
                    [[self.instructor_prompt, ngram] for ngram in ngram_batch]
                )
                embs.append(embs_batch)
            embs = np.vstack(embs).squeeze()

        else:
            embs = imodelsx.auggam.embed.embed_and_sum_function(
                ngrams_list,
                model=model,
                ngrams=None,
                tokenizer_embeddings=tokenizer_embeddings,
                tokenizer_ngrams=self.tokenizer_ngrams,
                checkpoint=self.checkpoint,
                layer=self.layer,
                batch_size=batch_size,
                # only return a single embedding
                fit_with_ngram_decomposition=False,
                sum_embeddings=False,
            )["embs"]
            embs = np.array(embs).squeeze()  # num_examples x embedding_size

        return _clean_np_array(embs)

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
                min_frequency=self.min_frequency,
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
            if preds.ndim > 1:  # multiclass classification
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

    def _predict_cached(self, X, warn):
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
        return np.array(preds)


class AugGAMRegressor(AugGAM, RegressorMixin):
    ...


class AugGAMClassifier(AugGAM, ClassifierMixin):
    ...


def _clean_np_array(arr):
    """Replace inf and nan with 0"""
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    return arr
