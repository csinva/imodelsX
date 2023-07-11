"""
Simple scikit-learn interface for finetuning a single linear layer on top of LLM embeddings.
"""
from numpy.typing import ArrayLike
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from spacy.lang.en import English
from scipy.sparse import issparse
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import transformers
from tqdm import tqdm
import os
import os.path
import warnings
import pickle as pkl
import torch
from sklearn.exceptions import ConvergenceWarning

device = "cuda" if torch.cuda.is_available() else "cpu"


class LinearNgram(BaseEstimator):
    def __init__(
        self,
        checkpoint: str = "tfidfvectorizer",
        tokenizer=None,
        ngrams=2,
        all_ngrams=True,
        random_state=None,
    ):
        """LinearNgram Class - use either LinearNgramClassifier or LinearNgramRegressor rather than initializing this class directly.

        Parameters
        ----------
        checkpoint: str
            Name of vectorizer checkpoint: "countvectorizer" or "tfidfvectorizer"
        ngrams
            Order of ngrams to extract. 1 for unigrams, 2 for bigrams, etc.
        all_ngrams
            Whether to use all order ngrams <= ngrams argument
        random_state
            random seed for fitting

        Example
        -------
        ```
        from imodelsx import LinearNgramClassifier
        import datasets
        import numpy as np

        # load data
        dset = datasets.load_dataset('rotten_tomatoes')['train']
        dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
        dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
        dset_val = dset_val.select(np.random.choice(len(dset_val), size=300, replace=False))


        # fit a simple ngram model
        m = LinearNgramClassifier()
        m.fit(dset['text'], dset['label'])
        preds = m.predict(dset_val['text'])
        acc = (preds == dset_val['label']).mean()
        print('validation acc', acc)
        ```
        """
        assert checkpoint in ["countvectorizer", "tfidfvectorizer"]
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.ngrams = ngrams
        self.all_ngrams = all_ngrams
        self.random_state = random_state

    def fit(
        self,
        X_text: ArrayLike,
        y: ArrayLike,
        verbose=True,
    ):
        """Extract embeddings then fit linear model

        Parameters
        ----------
        X_text: ArrayLike[str]
        y: ArrayLike[str]
        """

        # metadata
        if isinstance(self, ClassifierMixin):
            self.classes_ = unique_labels(y)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # set up model
        if verbose:
            print("initializing model...")

        # get embs
        if verbose:
            print("calculating embeddings...")
        if self.all_ngrams:
            lower_ngram = 1
        else:
            lower_ngram = self.ngrams

        # get vectorizer
        if self.checkpoint == "countvectorizer":
            self.vectorizer = CountVectorizer(
                tokenizer=self.tokenizer, ngram_range=(lower_ngram, self.ngrams)
            )
        elif self.checkpoint == "tfidfvectorizer":
            self.vectorizer = TfidfVectorizer(
                tokenizer=self.tokenizer, ngram_range=(lower_ngram, self.ngrams)
            )

        # get embs
        embs = self.vectorizer.fit_transform(X_text)

        # train linear
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if verbose:
            print("training linear model...")
        if isinstance(self, ClassifierMixin):
            self.linear = LogisticRegressionCV()
        elif isinstance(self, RegressorMixin):
            self.linear = RidgeCV()
        self.linear.fit(embs, y)

        return self

    def predict(self, X_text):
        """For regression returns continuous output.
        For classification, returns discrete output.
        """
        check_is_fitted(self)
        embs = self.vectorizer.transform(X_text)
        return self.linear.predict(embs)

    def predict_proba(self, X_text):
        check_is_fitted(self)
        embs = self.vectorizer.transform(X_text)
        return self.linear.predict_proba(embs)


class LinearNgramRegressor(LinearNgram, RegressorMixin):
    ...


class LinearNgramClassifier(LinearNgram, ClassifierMixin):
    ...


if __name__ == "__main__":
    import imodelsx.data

    dset, k = imodelsx.data.load_huggingface_dataset(
        "rotten_tomatoes", binary_classification=False, subsample_frac=0.1
    )
    print(dset)
    print(dset["train"])
    print(np.unique(dset["train"]["label"]))

    clf = LinearNgramClassifier()
    clf.fit(dset["train"]["text"], dset["train"]["label"])

    print("predicting")
    preds = clf.predict(dset["test"]["text"])
    print(preds.shape)

    print("predicting proba")
    preds_proba = clf.predict_proba(dset["test"]["text"])
    print(preds_proba.shape)

    assert preds_proba.shape[0] == preds.shape[0]
    print(
        "acc_train",
        np.mean(clf.predict(dset["train"]["text"]) == dset["train"]["label"]),
    )
    print("acc_test", np.mean(preds == dset["test"]["label"]))
