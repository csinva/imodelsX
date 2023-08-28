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
import transformers
from tqdm import tqdm
import os
import os.path
import warnings
import pickle as pkl
import torch
from sklearn.exceptions import ConvergenceWarning

device = "cuda" if torch.cuda.is_available() else "cpu"


class LinearFinetune(BaseEstimator):
    def __init__(
        self,
        checkpoint: str = "bert-base-uncased",
        layer: str = "last_hidden_state",
        random_state=None,
        normalize_embs=False,
        cache_embs_dir: str = None,
        verbose: int = 0,
    ):
        """LinearFinetune Class - use either LinearFinetuneClassifier or LinearFinetuneRegressor rather than initializing this class directly.

        Parameters
        ----------
        checkpoint: str
            Name of model checkpoint (i.e. to be fetch by huggingface)
        layer: str
            Name of layer to extract embeddings from
        random_state
            random seed for fitting
        normalize_embs
            whether to normalize embeddings before fitting linear model
        cache_embs_dir, optional
            if not None, directory to save embeddings into

        Example
        -------
        ```
        from imodelsx import LinearFinetuneClassifier
        import datasets
        import numpy as np

        # load data
        dset = datasets.load_dataset('rotten_tomatoes')['train']
        dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
        dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
        dset_val = dset_val.select(np.random.choice(len(dset_val), size=300, replace=False))


        # fit a simple one-layer finetune
        m = LinearFinetuneClassifier(
            checkpoint='distilbert-base-uncased',
        )
        m.fit(dset['text'], dset['label'])
        preds = m.predict(dset_val['text'])
        acc = (preds == dset_val['label']).mean()
        print('validation acc', acc)
        ```
        """
        self.checkpoint = checkpoint
        self.layer = layer
        self.random_state = random_state
        self.normalize_embs = normalize_embs
        self.cache_embs_dir = cache_embs_dir
        self.verbose = verbose
        self._initialize_checkpoint_and_tokenizer()

    def _initialize_checkpoint_and_tokenizer(self):
        self.model = transformers.AutoModel.from_pretrained(self.checkpoint).to(device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.checkpoint)

    def fit(
        self,
        X_text: ArrayLike,
        y: ArrayLike,
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
        if self.verbose:
            print("initializing model...")

        # get embs
        if self.verbose:
            print("calculating embeddings...")
        if self.cache_embs_dir is not None and os.path.exists(
            os.path.join(self.cache_embs_dir, "embs.pkl")
        ):
            embs = pkl.load(open(os.path.join(self.cache_embs_dir, "embs.pkl"), "rb"))
        else:
            embs = self._get_embs(X_text)
            if self.cache_embs_dir is not None:
                os.makedirs(self.cache_embs_dir, exist_ok=True)
                pkl.dump(
                    embs, open(os.path.join(self.cache_embs_dir, "embs.pkl"), "wb")
                )
        if self.normalize_embs:
            self.normalizer = StandardScaler()
            embs = self.normalizer.fit_transform(embs)

        # train linear
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if self.verbose:
            print("training linear model...")
        if isinstance(self, ClassifierMixin):
            self.linear = LogisticRegressionCV()
        elif isinstance(self, RegressorMixin):
            self.linear = RidgeCV()
        self.linear.fit(embs, y)

        return self

    def _get_embs(self, X_text: ArrayLike):
        embs = []
        if isinstance(X_text, list):
            n = len(X_text)
        else:
            n = X_text.shape[0]
        for i in tqdm(range(n)):
            inputs = self.tokenizer(
                [X_text[i]], padding="max_length", truncation=True, return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)
            output = self.model(**inputs)
            emb = output[self.layer].cpu().detach().numpy()
            if len(emb.shape) == 3:  # includes seq_len
                emb = emb.mean(axis=1)
            embs.append(emb)
        return np.array(embs).squeeze()  # num_examples x embedding_size

    def predict(self, X_text):
        """For regression returns continuous output.
        For classification, returns discrete output.
        """
        check_is_fitted(self)
        embs = self._get_embs(X_text)
        if self.normalize_embs:
            embs = self.normalizer.transform(embs)
        return self.linear.predict(embs)

    def predict_proba(self, X_text):
        check_is_fitted(self)
        embs = self._get_embs(X_text)
        if self.normalize_embs:
            embs = self.normalizer.transform(embs)
        return self.linear.predict_proba(embs)


class LinearFinetuneRegressor(LinearFinetune, RegressorMixin):
    ...


class LinearFinetuneClassifier(LinearFinetune, ClassifierMixin):
    ...


if __name__ == "__main__":
    import imodelsx.data

    dset, k = imodelsx.data.load_huggingface_dataset(
        "rotten_tomatoes", binary_classification=False, subsample_frac=0.1
    )
    print(dset)
    print(dset["train"])
    print(np.unique(dset["train"]["label"]))

    clf = LinearFinetuneClassifier()
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
