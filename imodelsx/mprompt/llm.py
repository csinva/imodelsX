import json
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain.cache import InMemoryCache
import re
from typing import Any, Dict, List, Mapping, Optional
import numpy as np
import openai
import os.path
from os.path import join, dirname
import os
import pickle as pkl
import langchain
from scipy.special import softmax
import openai
from langchain.llms.base import LLM
import hashlib
import torch
from mprompt.config import CACHE_DIR

# repo_dir = join(dirname(dirname(__file__)))
# langchain.llm_cache = InMemoryCache()

"""Wrapper class to call different language models
"""


def get_llm(checkpoint):
    if checkpoint.startswith("text-da") or "-00" in checkpoint:
        return llm_openai(checkpoint)
    elif checkpoint.startswith("gpt-3") or checkpoint.startswith("gpt-4"):
        return llm_openai_chat(checkpoint)
    else:
        return llm_hf(checkpoint)


def llm_openai(checkpoint="text-davinci-003") -> LLM:
    class LLM_OpenAI:
        def __init__(self, checkpoint, cache_dir=join(CACHE_DIR, checkpoint)):
            self.checkpoint = checkpoint
            self.cache_dir = cache_dir

        def __call__(self, prompt: str, max_new_tokens=250, seed=1, do_sample=True):
            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            hash_str = hashlib.sha256(prompt.encode()).hexdigest()
            cache_file = join(
                self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}__seed={seed}.pkl"
            )
            cache_file_raw = join(
                self.cache_dir,
                f"raw_{hash_str}__num_tok={max_new_tokens}__seed={seed}.pkl",
            )
            if os.path.exists(cache_file):
                return pkl.load(open(cache_file, "rb"))

            response = openai.Completion.create(
                engine=self.checkpoint,
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                # stop=["101"]
            )
            response_text = response["choices"][0]["text"]

            pkl.dump(response_text, open(cache_file, "wb"))
            pkl.dump(
                {"prompt": prompt, "response_text": response_text},
                open(cache_file_raw, "wb"),
            )
            return response_text

    return LLM_OpenAI(checkpoint)


def llm_openai_chat(checkpoint="gpt-3.5-turbo") -> LLM:
    class LLM_Chat:
        """Chat models take a different format: https://platform.openai.com/docs/guides/chat/introduction"""

        def __init__(self, checkpoint, cache_dir=join(CACHE_DIR, checkpoint)):
            self.checkpoint = checkpoint
            self.cache_dir = cache_dir

        def __call__(
            self,
            prompts_list: List[Dict[str, str]],
            max_new_tokens=250,
            seed=1,
            do_sample=True,
        ):
            """
            prompts_list: list of dicts, each dict has keys 'role' and 'content'
                Example: [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
            """
            assert isinstance(prompts_list, list), prompts_list

            # cache
            os.makedirs(self.cache_dir, exist_ok=True)
            prompts_list_dict = {
                str(i): sorted(v.items()) for i, v in enumerate(prompts_list)
            }
            if not self.checkpoint == "gpt-3.5-turbo":
                prompts_list_dict["checkpoint"] = self.checkpoint
            dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
            hash_str = hashlib.sha256(dict_as_str.encode()).hexdigest()
            cache_file_raw = join(
                self.cache_dir,
                f"chat__raw_{hash_str}__num_tok={max_new_tokens}__seed={seed}.pkl",
            )
            if os.path.exists(cache_file_raw):
                print("cached!")
                return pkl.load(open(cache_file_raw, "rb"))
            print("not cached")

            response = openai.ChatCompletion.create(
                model=self.checkpoint,
                messages=prompts_list,
                max_tokens=max_new_tokens,
                temperature=0.1,
                top_p=1,
                frequency_penalty=0.25,  # maximum is 2
                presence_penalty=0,
                # stop=["101"]
            )

            pkl.dump(response, open(cache_file_raw, "wb"))
            return response

    return LLM_Chat(checkpoint)


def _get_tokenizer(checkpoint):
    if "facebook/opt" in checkpoint:
        # opt can't use fast tokenizer
        # https://huggingface.co/docs/transformers/model_doc/opt
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    else:
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=True)


def llm_hf(checkpoint="google/flan-t5-xl") -> LLM:
    class LLM_HF:
        def __init__(self, checkpoint):
            _checkpoint: str = checkpoint
            self._tokenizer = _get_tokenizer(_checkpoint)
            if "google/flan" in checkpoint:
                self._model = T5ForConditionalGeneration.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16
                )
            elif checkpoint == "gpt-xl":
                self._model = AutoModelForCausalLM.from_pretrained(checkpoint)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    checkpoint, device_map="auto", torch_dtype=torch.float16
                )

        def __call__(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            max_new_tokens=20,
            do_sample=False,
        ) -> str:
            if stop is not None:
                raise ValueError("stop kwargs are not permitted.")
            inputs = self._tokenizer(
                prompt, return_tensors="pt", return_attention_mask=True
            ).to(
                self._model.device
            )  # .input_ids.to("cuda")
            # stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)])
            # outputs = self._model.generate(input_ids, max_length=max_tokens, stopping_criteria=stopping_criteria)
            # print('pad_token', self._tokenizer.pad_token)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                # pad_token=self._tokenizer.pad_token,
                pad_token_id=self._tokenizer.pad_token_id,
                # top_p=0.92,
                # top_k=0
            )
            out_str = self._tokenizer.decode(outputs[0])
            if "facebook/opt" in checkpoint:
                return out_str[len("</s>") + len(prompt) :]
            elif "google/flan" in checkpoint:
                print("full", out_str)
                return out_str[len("<pad>") : out_str.index("</s>")]
            else:
                return out_str[len(prompt) :]

        def _get_logit_for_target_token(
            self, prompt: str, target_token_str: str
        ) -> float:
            """Get logits target_token_str
            This is weird when token_output_ids represents multiple tokens
            It currently will only take the first token
            """
            # Get first token id in target_token_str
            target_token_id = self._tokenizer(target_token_str)["input_ids"][0]

            # get prob of target token
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
                padding=False,
                truncation=False,
            ).to(self._model.device)
            # shape is (batch_size, seq_len, vocab_size)
            logits = self._model(**inputs)["logits"].detach().cpu()
            # shape is (vocab_size,)
            probs_next_token = softmax(logits[0, -1, :].numpy().flatten())
            return probs_next_token[target_token_id]

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            """Get the identifying parameters."""
            return vars(self)

        @property
        def _llm_type(self) -> str:
            return "custom_hf_llm_for_langchain"

    return LLM_HF(checkpoint)


def get_paragraphs(
    prompts,
    checkpoint="gpt-4-0314",
    prefix_first="Write the beginning paragraph of a story about",
    prefix_next="Write the next paragraph of the story, but now make it about",
):
    """
    Example messages
    ----------------
    [
      {'role': 'system', 'content': 'You are a helpful assistant.'},
      {'role': 'user', 'content': 'Write the beginning paragraph of a story about "baseball". Make sure it contains several references to "baseball".'},
      {'role': 'assistant', 'content': 'The crack of the bat echoed through the stadium as the ball soared over the outfield fence. The crowd erupted into cheers, their excitement palpable. It was a beautiful day for baseball, with the sun shining down on the field and the smell of freshly cut grass filling the air. The players on the field were focused and determined, each one ready to give their all for their team. Baseball was more than just a game to them; it was a passion, a way of life. And as they took their positions on the field, they knew that anything was possible in this great game of baseball.'},
      {'role': 'user', 'content': 'Write the next paragraph of the story, but now make it about "animals". Make sure it contains several references to "animals".'},
    ]
    """
    token_limit = {
        "gpt-3.5-turbo": 3200,
        "gpt-4-0314": 30000,
    }[checkpoint]

    llm = get_llm(checkpoint)
    response = None
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    all_content = []
    for i in range(len(prompts)):
        messages.append({"role": "user", "content": prompts[i]})
        all_content.append(messages[-1])
        # for message in messages:
        # print(message)
        response = llm(messages)

        if response is not None:
            response_text = response["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": response_text})
            all_content.append(messages[-1])

        # need to drop beginning of story whenever we approach the tok limit
        # gpt-3.5.turbo has a limit of 4096, and it cant generate beyond that
        num_tokens = response["usage"]["total_tokens"]
        # print('num_tokens', num_tokens)
        if num_tokens >= token_limit:
            # drop the first (assistant, user) pair in messages
            messages = [messages[0]] + messages[3:]

            # rewrite the original prompt to now say beginning paragraph rather than next paragraph
            messages[1]["content"] = messages[1]["content"].replace(
                prefix_next, prefix_first
            )

    # extract out paragraphs
    paragraphs = [d["content"] for d in all_content if d["role"] == "assistant"]
    paragraphs
    assert len(paragraphs) == len(prompts)
    return paragraphs


if __name__ == "__main__":
    # llm = get_llm('text-davinci-003')
    # llm = get_llm('text-curie-001')
    llm = get_llm("text-ada-001")
    text = llm("Question: What do these have in common? Horse, cat, dog. Answer:")
    print("text", text)
