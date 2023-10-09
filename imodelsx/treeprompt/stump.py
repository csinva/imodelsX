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
        prompt_template: str = "{example}{prompt}",
        cache_key_values: bool = False,
        verbose: bool = True,
        model: AutoModelForCausalLM = None,
        checkpoint: str = "EleutherAI/gpt-j-6B",
        verbalizer: Dict[int, str] = {0: " Negative.", 1: " Positive."},
        batch_size: int = 1,
    ):
        """Given a prompt, extract its outputs

        Params
        ------
        args: contains some parameters passed through namespace (can ignore these)
        prompt: str
            the prompt to use (optional)
        prompt_template: str
            template for the prompt, for different prompt styles (e.g. few-shot), may want to place {prompt} before {example}
            or you may want to add some text before the verbalizer, e.g. {example}{prompt} Output:
        cache_key_values: bool
            Whether to cache key values (only possible when prompt does not start with {example})
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
        self.prompt_template = prompt_template
        self.cache_key_values = cache_key_values
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.model = model
        if tokenizer is None:
            self.tokenizer = imodelsx.llm.load_tokenizer(checkpoint)
        else:
            self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.verbalizer = verbalizer

        if self.verbose:
            logging.info(f"Loading model {self.checkpoint}")

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
        text_inputs = [self.prompt_template.format(
            example=x, prompt=self.prompt) for x in X_text]

        preds = self._get_logit_for_target_tokens_batched(
            text_inputs,
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

        text_inputs = [self.prompt_template.format(
            example=x, prompt=self.prompt) for x in X_text]

        preds = self._get_logit_for_target_tokens_batched_with_cache(
            past_key_values,
            text_inputs,
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

        p = self.prompt
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
        ):  # 'dbpedia' in self.args.dataset_name or max_len_prompt < 0: # dbpedia
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
        print(
            f"max_len_prompt: {max_len_prompt}, max_len_input: {max_len_input}")
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
                        logits[i, token_output_position,
                               token_output_id].item()
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
                        past_key_values[i][0].expand(
                            len(prompts_batch), -1, -1, -1),
                        past_key_values[i][1].expand(
                            len(prompts_batch), -1, -1, -1),
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
                outputs = self.model(
                    **inputs, past_key_values=past_key_values_new)
            logits = outputs["logits"]
            token_output_positions = (
                inputs["attention_mask"].sum(
                    axis=1) - past_key_values[0][0].shape[-2]
            )
            for i in range(len(prompts_batch)):
                token_output_position = token_output_positions[i].item() - 1
                logit_targets_list.append(
                    [
                        logits[i, token_output_position,
                               token_output_id].item()
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
