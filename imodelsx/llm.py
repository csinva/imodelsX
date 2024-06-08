from copy import deepcopy
import json
from transformers import (
    T5ForConditionalGeneration,
)
from datasets import Dataset
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
import re
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Any, Dict, List, Mapping, Optional, Union
import numpy as np
import os.path
from os.path import join, dirname
import os
import pickle as pkl
from scipy.special import softmax
import hashlib
import torch
from os.path import expanduser
import traceback
import time
from tqdm import tqdm

HF_TOKEN = None
if 'HF_TOKEN' in os.environ:
    HF_TOKEN = os.environ.get("HF_TOKEN")
elif os.path.exists(expanduser('~/.HF_TOKEN')):
    HF_TOKEN = open(expanduser('~/.HF_TOKEN'), 'r').read().strip()
if os.path.exists(expanduser('~/.OPENAI_API_KEY')):
    OPENAI_API_KEY = open(expanduser('~/.OPENAI_API_KEY'), 'r').read().strip()
if os.path.exists(expanduser('~/.OPENAI_API_KEY_SHARED')):
    OPENAI_API_KEY_SHARED = open(expanduser(
        '~/.OPENAI_API_KEY_SHARED'), 'r').read().strip()
'''
Example usage:
# gpt-4, gpt-35-turbo, meta-llama/Llama-2-70b-hf, mistralai/Mistral-7B-v0.1
checkpoint = 'meta-llama/Llama-2-7b-hf'
llm = imodelsx.llm.get_llm(checkpoint)
llm('may the force be') # returns ' with you'
'''

# change these settings before using these classes!
LLM_CONFIG = {
    # how long to wait before recalling a failed llm call (can set to None)
    "LLM_REPEAT_DELAY": None,
    "CACHE_DIR": join(
        os.path.expanduser("~"), "clin/CACHE_OPENAI"
    ),  # path to save cached llm outputs
    "LLAMA_DIR": join(
        os.path.expanduser("~"), "llama"
    ),  # path to extracted llama weights
}


def get_llm(
    checkpoint,
    seed=1,
    role: str = None,
    repeat_delay: Optional[float] = None,
    CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
    LLAMA_DIR=LLM_CONFIG["LLAMA_DIR"],
):
    if repeat_delay is not None:
        LLM_CONFIG["LLM_REPEAT_DELAY"] = repeat_delay

    """Get an LLM with a call function and caching capabilities"""
    if checkpoint.startswith("gpt-3") or checkpoint.startswith("gpt-4"):
        return LLM_Chat(checkpoint, seed, role, CACHE_DIR)
    elif 'Meta-Llama-3' in checkpoint and 'Instruct' in checkpoint:
        return LLM_HF_Pipeline(checkpoint, CACHE_DIR)
    else:
        # warning: this sets torch.manual_seed(seed)
        return LLM_HF(checkpoint, seed=seed, CACHE_DIR=CACHE_DIR, LLAMA_DIR=LLAMA_DIR)


def repeatedly_call_with_delay(llm_call):
    def wrapper(*args, **kwargs):
        # Number of seconds to wait between calls (None will not repeat)
        delay = LLM_CONFIG["LLM_REPEAT_DELAY"]
        response = None
        while response is None:
            try:
                response = llm_call(*args, **kwargs)

                # fix for when this function was returning response rather than string
                # if response is not None and not isinstance(response, str):
                # response = response["choices"][0]["message"]["content"]
            except Exception as e:
                e = str(e)
                print(e)
                if "does not exist" in e:
                    return None
                elif "maximum context length" in e:
                    return None
                elif 'content management policy' in e:
                    return None
                if delay is None:
                    raise e
                else:
                    time.sleep(delay)
        return response

    return wrapper


class LLM_Chat:
    """Chat models take a different format: https://platform.openai.com/docs/guides/chat/introduction"""

    def __init__(self, checkpoint, seed, role, CACHE_DIR):
        self.cache_dir = join(
            CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
        )
        self.checkpoint = checkpoint
        self.role = role
        from openai import AzureOpenAI
        if 'spot' in checkpoint:
            self.client = AzureOpenAI(
                azure_endpoint="https://gcraoai9wus3spot.openai.azure.com/",
                api_version="2024-02-01",
                api_key=OPENAI_API_KEY_SHARED,
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint="https://healthcare-ai.openai.azure.com/",
                api_version="2024-02-01",
                api_key=OPENAI_API_KEY,
            )

    # @repeatedly_call_with_delay
    def __call__(
        self,
        prompts_list: List[Dict[str, str]],
        max_new_tokens=250,
        stop=None,
        functions: List[Dict] = None,
        return_str=True,
        verbose=True,
        temperature=0.1,
        frequency_penalty=0.25,
        use_cache=True,
        return_false_if_not_cached=False,
    ):
        """
        prompts_list: list of dicts, each dict has keys 'role' and 'content'
            Example: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant",
                    "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        prompts_list: str
            Alternatively, string which gets formatted into basic prompts_list:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": <<<<<prompts_list>>>>},
            ]
        """
        if isinstance(prompts_list, str):
            role = self.role
            if role is None:
                role = "You are a helpful assistant."
            prompts_list = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompts_list},
            ]

        assert isinstance(prompts_list, list), prompts_list

        # cache
        os.makedirs(self.cache_dir, exist_ok=True)
        prompts_list_dict = {
            str(i): sorted(v.items()) for i, v in enumerate(prompts_list)
        }
        if not self.checkpoint == "gpt-3.5-turbo":
            prompts_list_dict["checkpoint"] = self.checkpoint
        if functions is not None:
            prompts_list_dict["functions"] = functions
        if temperature > 0.1:
            prompts_list_dict["temperature"] = temperature
        dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
        hash_str = hashlib.sha256(dict_as_str.encode()).hexdigest()
        cache_file = join(
            self.cache_dir,
            f"chat__{hash_str}__num_tok={max_new_tokens}.pkl",
        )
        if os.path.exists(cache_file) and use_cache:
            if verbose:
                print("cached!")
                # print(cache_file)
            # print(cache_file)
            response = pkl.load(open(cache_file, "rb"))
            if response is not None:
                return response
        if verbose:
            print("not cached")

        if return_false_if_not_cached:
            return False

        kwargs = dict(
            model=self.checkpoint,
            messages=prompts_list,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=frequency_penalty,  # maximum is 2
            presence_penalty=0,
            stop=stop,
            # logprobs=True,
            # stop=["101"]
        )
        # print('kwargs', kwargs)
        if functions is not None:
            kwargs["functions"] = functions

        response = self.client.chat.completions.create(
            **kwargs,
        )

        if return_str:
            response = response.choices[0].message.content

        if response is not None:
            # print('resp', response, 'cache_file', cache_file)
            try:
                pkl.dump(response, open(cache_file, "wb"))
            except:
                print('failed to save cache!', cache_file)
                traceback.print_exc()

        return response


def load_tokenizer(checkpoint: str) -> transformers.PreTrainedTokenizer:
    if "facebook/opt" in checkpoint:
        # opt can't use fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_fast=False, padding_side='left', token=HF_TOKEN)
    elif "PMC_LLAMA" in checkpoint:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "chaoyi-wu/PMC_LLAMA_7B", padding_side='left', token=HF_TOKEN)
    else:
        # , use_fast=True)
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, padding_side='left', use_fast=True, token=HF_TOKEN)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_hf_model(checkpoint: str) -> transformers.PreTrainedModel:
    # set checkpoint
    kwargs = {
        "pretrained_model_name_or_path": checkpoint,
        "output_hidden_states": False,
        # "pad_token_id": tokenizer.eos_token_id,
        "low_cpu_mem_usage": True,
    }
    if "google/flan" in checkpoint:
        return T5ForConditionalGeneration.from_pretrained(
            checkpoint, device_map="auto", torch_dtype=torch.float16
        )
    elif checkpoint == "EleutherAI/gpt-j-6B":
        return AutoModelForCausalLM.from_pretrained(
            checkpoint,
            revision="float16",
            torch_dtype=torch.float16,
            **kwargs,
        )
    elif "llama-2" in checkpoint.lower():
        return AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            device_map="auto",
            token=HF_TOKEN,
            offload_folder="offload",
        )
    elif "llama_" in checkpoint:
        return transformers.LlamaForCausalLM.from_pretrained(
            join(LLAMA_DIR, checkpoint),
            device_map="auto",
            torch_dtype=torch.float16,
        )
    elif 'microsoft/phi' in checkpoint:
        return AutoModelForCausalLM.from_pretrained(
            checkpoint
        )
    elif checkpoint == "gpt-xl":
        return AutoModelForCausalLM.from_pretrained(checkpoint)
    else:
        return AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map="auto", torch_dtype=torch.float16,
            token=HF_TOKEN,
        )


class LLM_HF_Pipeline:
    def __init__(self, checkpoint, CACHE_DIR):

        self.pipeline_ = transformers.pipeline(
            "text-generation",
            model=checkpoint,
            # model_kwargs={"torch_dtype": torch.bfloat16},
            # , 'device_map': "auto"},
            model_kwargs={'torch_dtype': torch.float16},
            device_map="auto"
        )
        self.pipeline_.tokenizer.pad_token_id = self.pipeline_.tokenizer.eos_token_id
        self.pipeline_.tokenizer.padding_side = 'left'
        # self.pipeline_.model.generation_config.pad_token_id = self.pipeline_.tokenizer.pad_token_id
        self.cache_dir = join(CACHE_DIR)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens=20,
        use_cache=True,
        verbose=False,
        batch_size=64,
    ):

        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            hash_str = hashlib.sha256(str(prompt).encode()).hexdigest()
            cache_file = join(
                self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl"
            )

            if os.path.exists(cache_file):
                if verbose:
                    print("cached!")
                try:
                    return pkl.load(open(cache_file, "rb"))
                except:
                    print('failed to load cache so rerunning...')
            if verbose:
                print("not cached...")
        outputs = self.pipeline_(
            prompt,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            do_sample=False,
            pad_token_id=self.pipeline_.tokenizer.pad_token_id,
        )
        if isinstance(prompt, str):
            texts = outputs[0]["generated_text"][len(prompt):]
        else:
            texts = [outputs[i][0]['generated_text']
                     [len(prompt[i]):] for i in range(len(outputs))]

        if use_cache:
            pkl.dump(texts, open(cache_file, "wb"))
        return texts


class LLM_HF:
    def __init__(self, checkpoint, seed, CACHE_DIR, LLAMA_DIR=None):
        self.tokenizer_ = load_tokenizer(checkpoint)
        self.model_ = load_hf_model(checkpoint)
        self.checkpoint = checkpoint
        self.cache_dir = join(
            CACHE_DIR, "cache_hf", f'{checkpoint.replace("/", "_")}___{seed}'
        )
        self.seed = seed

    def __call__(
        self,
        prompt: Union[str, List[str]],
        stop: str = None,
        max_new_tokens=20,
        do_sample=False,
        use_cache=True,
        verbose=False,
        return_next_token_prob_scores=False,
        target_token_strs: List[str] = None,
        return_top_target_token_str: bool = False,
        batch_size=1,
    ) -> Union[str, List[str]]:
        """Warning: stop is used posthoc but not during generation.
        Be careful, caching can take up a lot of memory....

        Example mistral-instruct prompt: "<s>[INST]'Input text: {example}\nQuestion: {question} Answer yes or no.[/INST]"


        Params
        ------
        return_next_token_prob_scores: bool
            If this is true, then the function will return the probability of the next token being each of the target_token_strs
            target_token_strs: List[str]
                If this is not None and return_next_token_prob_scores is True, then the function will return the probability of the next token being each of the target_token_strs
                The output will be a list of dictionaries in this case List[Dict[str, float]]
                return_top_target_token_str: bool
                    If true and above are true, then just return top token of the above
                    This is a way to constrain the output (but only for 1 token)
                    This setting caches but the other two (which do not return strings) do not cache

        """
        input_is_str = isinstance(prompt, str)
        with torch.no_grad():
            # cache
            if use_cache:
                os.makedirs(self.cache_dir, exist_ok=True)
                hash_str = hashlib.sha256(str(prompt).encode()).hexdigest()
                cache_file = join(
                    self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl"
                )

                if os.path.exists(cache_file):
                    if verbose:
                        print("cached!")
                    try:
                        return pkl.load(open(cache_file, "rb"))
                    except:
                        print('failed to load cache so rerunning...')
                if verbose:
                    print("not cached...")

            # if stop is not None:
            # raise ValueError("stop kwargs are not permitted.")
            inputs = self.tokenizer_(
                prompt, return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                truncation=False,
            ).to(self.model_.device)

            if return_next_token_prob_scores or target_token_strs or return_top_target_token_str:
                outputs = self.model_.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=self.tokenizer_.pad_token_id,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                next_token_logits = outputs['logits'][0]
                next_token_probs = next_token_logits.softmax(
                    axis=-1).detach().cpu().numpy()

                if target_token_strs is None:
                    return next_token_probs

                target_token_ids = self._check_target_token_strs(
                    target_token_strs)
                if return_top_target_token_str:
                    selected_tokens = next_token_probs[:, np.array(
                        target_token_ids)].squeeze().argmax(axis=-1)
                    out_strs = [
                        target_token_strs[selected_tokens[i]]
                        for i in range(len(selected_tokens))
                    ]
                    if len(out_strs) == 1:
                        out_strs = out_strs[0]
                    if use_cache:
                        pkl.dump(out_strs, open(cache_file, "wb"))
                    return out_strs
                else:
                    out_dict_list = [
                        {target_token_strs[i]: next_token_probs[prompt_num, target_token_ids[i]]
                            for i in range(len(target_token_strs))
                         }
                        for prompt_num in range(len(prompt))
                    ]
                    return out_dict_list
            else:
                outputs = self.model_.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer_.pad_token_id,
                )
                # top_p=0.92,
                # temperature=0,
                # top_k=0
            if input_is_str:
                out_str = self.tokenizer_.decode(
                    outputs[0], skip_special_tokens=True)
                # print('out_str', out_str)
                if 'mistral' in self.checkpoint and 'Instruct' in self.checkpoint:
                    out_str = out_str[len(prompt) - 2:]
                elif 'Meta-Llama-3' in self.checkpoint and 'Instruct' in self.checkpoint:
                    out_str = out_str[len(prompt) - 145:]
                else:
                    out_str = out_str[len(prompt):]

                if use_cache:
                    pkl.dump(out_str, open(cache_file, "wb"))
                return out_str
            else:
                out_strs = []
                for i in range(outputs.shape[0]):
                    out_tokens = outputs[i]
                    out_str = self.tokenizer_.decode(
                        out_tokens, skip_special_tokens=True)
                    if 'mistral' in self.checkpoint and 'Instruct' in self.checkpoint:
                        out_str = out_str[len(prompt[i]) - 2:]
                    elif 'Meta-Llama-3' in self.checkpoint and 'Instruct' in self.checkpoint:
                        # print('here')
                        out_str = out_str[len(prompt) + 187:]
                    else:
                        out_str = out_str[len(prompt[i]):]
                    out_strs.append(out_str)
                if use_cache:
                    pkl.dump(out_strs, open(cache_file, "wb"))
                return out_strs

    def _check_target_token_strs(self, target_token_strs, override_token_with_first_token_id=False):
        if isinstance(target_token_strs, str):
            target_token_strs = [target_token_strs]

        target_token_ids = [self.tokenizer_(target_token_str, add_special_tokens=False)["input_ids"]
                            for target_token_str in target_token_strs]

        # Check that the target token is in the vocab
        if override_token_with_first_token_id:
            # Get first token id in target_token_str
            target_token_ids = [target_token_id[0]
                                for target_token_id in target_token_ids]
        else:
            for i in range(len(target_token_strs)):
                if len(target_token_ids[i]) > 1:
                    raise ValueError(
                        f"target_token_str {target_token_strs[i]} has multiple tokens: " +
                        str([self.tokenizer_.decode(target_token_id)
                            for target_token_id in target_token_ids[i]]))
        return target_token_ids


class LLMEmbs:
    def __init__(self, checkpoint):
        self.tokenizer_ = load_tokenizer(checkpoint)
        self.model_ = AutoModel.from_pretrained(
            checkpoint, output_hidden_states=True,
            device_map="auto",
            torch_dtype=torch.float16,)

    def __call__(self, texts: List[str], layer_idx: int = 18, batch_size=16):
        '''Returns embeddings
        '''
        embs = []
        for i in tqdm(range(0, len(texts), batch_size)):
            inputs = self.tokenizer_(
                texts[i:i + batch_size], return_tensors='pt', padding=True).to(self.model_.device)
            hidden_states = self.model_(**inputs).hidden_states

            # layers x batch x tokens x features
            emb = hidden_states[layer_idx].detach().cpu().numpy()

            # get emb from last token
            emb = emb[:, -1, :]
            embs.append(deepcopy(emb))
        embs = np.concatenate(embs)
        return embs


if __name__ == "__main__":
    # llm = get_llm("text-davinci-003")
    # text = llm("What do these have in common? Horse, ")
    # print("text", text)

    # llm = get_llm("gpt2")
    # text = llm(
    # """Continue this list
    # - apple
    # - banana
    # -"""
    # )
    # print("text", text)
    # tokenizer = transformers.LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")
    # model = transformers.LlamaForCausalLM.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")

    # llm = get_llm("chaoyi-wu/PMC_LLAMA_7B")
    #     llm = get_llm("llama_65b")
    #     text = llm(
    #         """Continue this list
    # - red
    # - orange
    # - yellow
    # - green
    # -""",
    #         use_cache=False,
    #     )
    #     print(text)
    #     print("\n\n")
    #     print(repr(text))

    # GET LOGITS ###################################
    # llm = get_llm("gpt2")
    # prompts = ['roses are red, violets are', 'may the force be with']
    # # prompts = ['may the force be with', 'so may the light be with']
    # target_token_strs = [' blue', ' you']
    # ans = llm(prompts, return_next_token_prob_scores=True,
    #           use_cache=False, target_token_strs=target_token_strs)

    # FORCE WORDS ##########
    llm = get_llm("gpt2")
    prompts = ['roses are red, violets are',
               'may the force be with', 'trees are usually']
    # prompts = ['may the force be with', 'so may the light be with']
    target_token_strs = [' green', ' you', 'orange']
    llm._check_target_token_strs(target_token_strs)
    ans = llm(prompts, use_cache=False,
              return_next_token_prob_scores=True, target_token_strs=target_token_strs,
              return_top_target_token_str=True)
    print('ans', ans)
