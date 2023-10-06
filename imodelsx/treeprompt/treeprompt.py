import os
import pandas as pd
import numpy as np
import transformers
import sys
from os.path import join
import datasets
from typing import Dict, List
from dict_hash import sha256
import joblib
import random
import numpy as np
import sklearn.tree
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder

import imodelsx.treeprompt.stump


class TreePromptClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        checkpoint: str,
        prompts: List[str],
        verbalizer: Dict[int, str] = {0: " Negative.", 1: " Positive."},
        tree_kwargs: Dict = {"max_leaf_nodes": 3},
        batch_size: int = 1,
        cache_prompt_features_dir=join("cache_prompt_features"),
        cache_key_values: bool = False,
        device=None,
        verbose: bool = True,
        random_state: int = 42,
    ):
        self.checkpoint = checkpoint
        self.prompts = prompts
        self.verbalizer = verbalizer
        self.tree_kwargs = tree_kwargs
        self.batch_size = batch_size
        self.cache_prompt_features_dir = cache_prompt_features_dir
        self.cache_key_values = cache_key_values
        self.device = device
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y):
        transformers.set_seed(self.random_state)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        assert len(self.classes_) == len(self.verbalizer)

        # calculate prompt features
        prompt_features = self._calc_prompt_features(X, self.prompts)
        self.prompt_accs_ = [
            accuracy_score(y, prompt_features[:, i]) for i in range(len(self.prompts))
        ]

        # apply one-hot encoding to features
        if len(np.unique(y)) > 3:
            print("Converting to one-hot")
            self.enc_ = OneHotEncoder(handle_unknown="ignore")
            prompt_features = enc.fit_transform(prompt_features)
            self.feature_names_ = enc.get_feature_names_out(self.prompts)
        else:
            self.feature_names_ = self.prompts

        # train decision tree
        self.clf_ = sklearn.tree.DecisionTreeClassifier(
            **self.tree_kwargs,
            random_state=self.random_state,
        )
        self.clf_.fit(prompt_features, y)
        self.prompts_idxs_kept = np.unique(self.clf_.tree_.feature)[
            1:
        ]  # remove first element which is -2

        return self

    def _calc_prompt_features(self, X, prompts):
        prompt_features = np.zeros((len(X), len(prompts)))
        llm = imodelsx.llm.get_llm(self.checkpoint)._model
        if self.device is not None:
            llm = llm.to(self.device)
        stump = None

        for i, prompt in enumerate(prompts):
            print(f"Prompt {i}: {prompt}")

            loaded_from_cache = False
            if self.cache_prompt_features_dir is not None:
                os.makedirs(self.cache_prompt_features_dir, exist_ok=True)
                args_dict_cache = {"prompt": prompt}
                save_dir_unique_hash = sha256(args_dict_cache)
                cache_file = join(
                    self.cache_prompt_features_dir, f"{save_dir_unique_hash}.pkl"
                )

                # load from cache if possible
                if os.path.exists(cache_file):
                    print("loading from cache!")
                    try:
                        prompt_features_i = joblib.load(cache_file)
                        loaded_from_cache = True
                    except:
                        pass

            if not loaded_from_cache:
                if stump is None:
                    stump = imodelsx.treeprompt.stump.PromptStump(
                        model=llm,
                        checkpoint=self.checkpoint,
                        verbalizer=self.verbalizer,
                        batch_size=self.batch_size,
                    )

                # calculate prompt_features
                def _calc_features_single_prompt(
                    X, stump, prompt, past_key_values=None
                ):
                    """Calculate features with a single prompt (results get cached)
                    preds: np.ndarray[int] of shape (X.shape[0],)
                        If multiclass, each int takes value 0, 1, ..., n_classes - 1 based on the verbalizer
                    """
                    stump.prompt = prompt
                    if past_key_values is not None:
                        preds = stump.predict_with_cache(X, past_key_values)
                    else:
                        preds = stump.predict(X)
                    return preds

                past_key_values = None
                if self.cache_key_values:
                    stump.prompt = prompt
                    past_key_values = stump.calc_key_values(X_train_text)
                prompt_features_i = _calc_features_single_prompt(
                    X, stump, prompt, past_key_values=past_key_values
                )
                if self.cache_prompt_features_dir is not None:
                    joblib.dump(prompt_features_i, cache_file)

            # save prompt features
            prompt_features[:, i] = prompt_features_i

        return prompt_features

    def predict_proba(self, X):
        # extract prompt features
        prompt_features = np.zeros((len(X), len(self.prompts)))
        prompt_features_relevant = self._calc_prompt_features(
            X, np.array(self.prompts)[self.prompts_idxs_kept]
        )
        prompt_features[:, self.prompts_idxs_kept] = prompt_features_relevant

        # apply one-hot encoding to features
        if hasattr(self, "enc_"):
            X = self.enc_.transform(prompt_features)

        # predict
        return self.clf_.predict_proba(prompt_features)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


if __name__ == "__main__":
    # load data
    rng = np.random.default_rng(42)
    dset_train = datasets.load_dataset("rotten_tomatoes")["train"]
    dset_train = dset_train.select(rng.choice(
        len(dset_train), size=100, replace=False))
    dset_val = datasets.load_dataset("rotten_tomatoes")["validation"]
    dset_val = dset_val.select(rng.choice(
        len(dset_val), size=100, replace=False))

    # example fit
    prompts = [
        "This movie is",
        " Positive or Negative? The movie was",
        " The sentiment of the movie was",
        " The plot of the movie was really",
        " The acting in the movie was",
    ]
    verbalizer = {0: " Negative.", 1: " Positive."}
    m = TreePromptClassifier(
        checkpoint="gpt2",
        prompts=prompts,
        verbalizer=verbalizer,
        cache_prompt_features_dir=None,  # 'cache_prompt_features_dir/gp2',
    )
    m.fit(dset_train["text"], dset_train["label"])

    acc_test = np.mean(m.predict(dset_val["text"]) == dset_val["label"])
    print("acc_test", acc_test)

    # visualize decision tree
    # plot_tree(
    #     m.clf_,
    #     fontsize=10,
    #     feature_names=m.feature_names_,
    #     class_names=list(verbalizer.values()),
    #     filled=True,
    # )
    # plt.show()

    # pd.DataFrame({'prompt': prompts, 'acc': m.prompt_accs_})
