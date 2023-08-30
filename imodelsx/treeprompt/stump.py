from typing import Dict, List

from abc import ABC, abstractmethod
import logging
import math
import random
import imodels
import imodelsx.util
import imodelsx.metrics
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.cuda
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class PromptStump:
    def __init__(
        self,
        args=None,
        prompt: str = None,
        tokenizer=None,
        max_features=10,
        assert_checks: bool = False,
        verbose: bool = True,
        model: AutoModelForCausalLM = None,
        checkpoint: str = "EleutherAI/gpt-j-6B",
        verbalizer: Dict[int, str] = {0: " Negative.", 1: " Positive."},
        batch_size: int = 1,
    ):
        """Fit a single stump.
        Can use tabular features...
            Currently only supports binary classification with binary features.
        Params
        ------
        args: contains some parameters passed through namespace
        prompt: str
            the prompt to use (optional)
        max_features: int
            used by StumpTabular to decide how many features to save
        checkpoint: str
            the underlying model used for prediction
        model: AutoModelForCausalLM
            if this is passed, will override checkpoint
        """
        if args is None:

            class placeholder:
                prompt_source = None
                template_data_demonstrations = None
                dataset_name = ""

            self.args = placeholder()
        else:
            self.args = args
        self.prompt = prompt
        self.assert_checks = assert_checks
        self.verbose = verbose
        self.max_features = max_features
        self.checkpoint = checkpoint
        self.model = model
        if tokenizer is None:
            self.tokenizer = imodelsx.llm.load_tokenizer(checkpoint)
        else:
            self.tokenizer = tokenizer
        self.batch_size = batch_size

        # tree stuff
        self.child_left = None
        self.child_right = None
        self.verbalizer = verbalizer

        if self.verbose:
            logging.info(f"Loading model {self.checkpoint}")

    def fit(self, X_text: List[str], y, feature_names=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, "y should have more than 1 unique value"
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # actually run fitting
        input_strings = X_text
        output_strings = [self.verbalizer[int(yi)] for yi in y]

        # set value (calls self.predict, which uses self.prompt)
        self._set_value_acc_samples(X_text, y)

        return self

    def __getstate__(self):
        """Get the stump but prevent certain attributes from being pickled.

        See also https://stackoverflow.com/a/54139237/2287177
        """
        state = self.__dict__.copy()
        # Don't pickle big things
        if "model" in state:
            del state["model"]
        if "tokenizer" in state:
            del state["tokenizer"]
        if "feature_names" in state:
            del state["feature_names"]
        return state

    def predict(self, X_text: List[str]) -> np.ndarray[int]:
        preds_proba = self.predict_proba(X_text)
        return np.argmax(preds_proba, axis=1)

    def predict_with_cache(self, X_text: List[str], past_key_values) -> np.ndarray[int]:
        preds_proba = self.predict_proba_with_cache(X_text, past_key_values)
        return np.argmax(preds_proba, axis=1)

    def predict_proba(self, X_text: List[str]) -> np.ndarray[float]:
        target_strs = list(self.verbalizer.values())

        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        assert len(set(target_token_ids)) == len(
            set(target_strs)
        ), f"error: target_token_ids {set(target_token_ids)} not unique to target strings {set(target_strs)}"

        if self.args.prompt_source == "data_demonstrations":
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_target_tokens_batched(
                [self.prompt + template % (x, "") for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        else:
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def predict_proba_with_cache(
        self, X_text: List[str], past_key_values
    ) -> np.ndarray[float]:
        target_strs = list(self.verbalizer.values())

        # only predict based on first token of output string
        target_token_ids = list(map(self._get_first_token_id, target_strs))
        assert len(set(target_token_ids)) == len(
            set(target_strs)
        ), f"error: target_token_ids {set(target_token_ids)} not unique to target strings {set(target_strs)}"

        if self.args.prompt_source == "data_demonstrations":
            template = self.args.template_data_demonstrations
            preds = self._get_logit_for_target_tokens_batched_with_cache(
                past_key_values,
                [template % (x, "") for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        else:
            raise NotImplementedError
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def calc_key_values(self, X_text: List[str]):
        # only predict based on first token of output string
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding = True

        self.tokenizer.pad_token = self.tokenizer.eos_token
        if (
            self.args.prompt_source == "data_demonstrations"
            or self.args.prompt_source == "data_demonstrations_knn"
        ):
            p = self.prompt
            if self.args.prompt_source == "data_demonstrations_knn":
                p = self.prompt[0]
            template = self.args.template_data_demonstrations
            if self.args.dataset_name.startswith("knnp__"):
                max_len_verb = max(
                    len(self.tokenizer.encode(v)) for v in self.verbalizer.values()
                )
                max_len_input = (
                    max_len_verb
                    + max(len(self.tokenizer.encode(s)) for s in X_text)
                    + 1
                )
            else:
                max_len_input = -1
                for v in self.verbalizer.values():
                    max_len_input = max(
                        max_len_input,
                        max(
                            [
                                len(self.tokenizer.encode(template % (s, v)))
                                for s in X_text[:1000]
                            ]
                        ),
                    )
            try:
                max_total_len = self.model.config.n_positions
            except:
                max_total_len = self.model.config.max_position_embeddings
            max_len_prompt = max_total_len - max_len_input
            if (
                True
            ):  #'dbpedia' in self.args.dataset_name or max_len_prompt < 0: # dbpedia
                print("max len prompt less than 0, truncating to the left")
                max_len_input = -1
                for v in self.verbalizer.values():
                    a = [
                        len(self.tokenizer.encode(template % (s, v)))
                        for s in X_text[:1000]
                    ]
                    max_len_input = max(max_len_input, np.percentile(a, 95))
            max_len_input = int(math.ceil(max_len_input))
            max_len_prompt = max_total_len - max_len_input
            self.max_len_input = max_len_input
            print(f"max_len_prompt: {max_len_prompt}, max_len_input: {max_len_input}")
            assert max_len_prompt > 0
            inputs = self.tokenizer(
                [
                    p,
                ],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len_prompt,
                return_attention_mask=True,
            ).to(self.model.device)

            # shape is (batch_size, seq_len, vocab_size)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs["past_key_values"]
        else:
            raise NotImplementedError
            preds = self._get_logit_for_target_tokens_batched(
                [x + self.prompt for x in X_text],
                target_token_ids,
                batch_size=self.batch_size,
            )
        assert preds.shape == (len(X_text), len(target_token_ids)), (
            "preds shape was"
            + str(preds.shape)
            + " but should have been "
            + str((len(X_text), len(target_token_ids)))
        )

        # return the class with the highest logit
        return softmax(preds, axis=1)

    def _get_logit_for_target_tokens_batched(
        self, prompts: List[str], target_token_ids: List[int], batch_size: int = 64
    ) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        try:
            max_total_len = self.model.config.n_positions
        except:
            max_total_len = self.model.config.max_position_embeddings

        pbar = tqdm.tqdm(
            total=len(prompts),
            leave=False,
            desc="getting dataset predictions for top prompt",
            colour="red",
        )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return np.array(logit_targets_list)

            prompts_batch = prompts[batch_start:batch_end]
            self.tokenizer.padding = True
            self.tokenizer.truncation_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True,
                max_length=max_total_len,
            ).to(self.model.device)

            # shape is (batch_size, seq_len, vocab_size)
            with torch.no_grad():
                logits = self.model(**inputs)["logits"]

            token_output_positions = inputs["attention_mask"].sum(axis=1)
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(
                    [
                        logits[i, token_output_position, token_output_id].item()
                        for token_output_id in target_token_ids
                    ]
                )

    def _get_logit_for_target_tokens_batched_with_cache(
        self,
        past_key_values,
        prompts: List[str],
        target_token_ids: List[int],
        batch_size: int = 64,
    ) -> np.ndarray[float]:
        """Get logits for each target token
        This can fail when token_output_ids represents multiple tokens
        So things get mapped to the same id representing "unknown"
        """
        logit_targets_list = []
        batch_num = 0

        pbar = tqdm.tqdm(
            total=len(prompts), leave=False, desc="getting predictions", colour="red"
        )

        past_key_values_new = []
        for i in range(len(past_key_values)):
            past_key_values_new.append(
                [
                    past_key_values[i][0].expand(batch_size, -1, -1, -1),
                    past_key_values[i][1].expand(batch_size, -1, -1, -1),
                ]
            )
        while True:
            batch_start = batch_num * batch_size
            batch_end = (batch_num + 1) * batch_size
            batch_num += 1
            pbar.update(batch_size)
            if batch_start >= len(prompts):
                return np.array(logit_targets_list)

            prompts_batch = prompts[batch_start:batch_end]
            if len(prompts_batch) != past_key_values_new[0][0].shape[0]:
                for i in range(len(past_key_values)):
                    past_key_values_new[i] = [
                        past_key_values[i][0].expand(len(prompts_batch), -1, -1, -1),
                        past_key_values[i][1].expand(len(prompts_batch), -1, -1, -1),
                    ]
            self.tokenizer.padding = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len_input,
                return_attention_mask=True,
            ).to(self.model.device)

            attention_mask = inputs["attention_mask"]
            attention_mask = torch.cat(
                (
                    attention_mask.new_zeros(
                        len(prompts_batch), past_key_values[0][0].shape[-2]
                    ).fill_(1),
                    attention_mask,
                ),
                dim=-1,
            )
            inputs["attention_mask"] = attention_mask

            # shape is (batch_size, seq_len, vocab_size)
            with torch.no_grad():
                outputs = self.model(**inputs, past_key_values=past_key_values_new)
            logits = outputs["logits"]
            token_output_positions = (
                inputs["attention_mask"].sum(axis=1) - past_key_values[0][0].shape[-2]
            )
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(
                    [
                        logits[i, token_output_position, token_output_id].item()
                        for token_output_id in target_token_ids
                    ]
                )

    def _get_first_token_id(self, prompt: str) -> str:
        """Get first token id in prompt (after special tokens).

        Need to strip special tokens for LLAMA so we don't get a special space token at the beginning.
        """
        if "llama" in self.checkpoint.lower():
            prompt = prompt.lstrip()

        tokens = self.tokenizer(prompt)["input_ids"]
        tokens = [t for t in tokens if t not in self.tokenizer.all_special_ids]
        return tokens[0]

    def __str__(self):
        return f"PromptStump(val={self.value_mean:0.2f} n={np.sum(self.n_samples)} prompt={self.prompt})"

    def get_str_simple(self):
        return self.prompt

    def _set_value_acc_samples(self, X_text, y):
        """Set value and accuracy of stump."""
        idxs_right = self.predict(X_text).astype(bool)
        n_right = idxs_right.sum()
        if n_right == 0 or n_right == y.size:
            self.failed_to_split = True
            return
        else:
            self.failed_to_split = False
        self.value = [np.mean(y[~idxs_right]), np.mean(y[idxs_right])]
        self.value_mean = np.mean(y)
        self.n_samples = [y.size - idxs_right.sum(), idxs_right.sum()]
        self.acc = accuracy_score(y, 1 * idxs_right)
