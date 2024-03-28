import json
from transformers import (
    T5ForConditionalGeneration,
)
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
import time


'''
Example usage:
checkpoint = 'meta-llama/Llama-2-7b-hf' # gpt-4, gpt-35-turbo, meta-llama/Llama-2-70b-hf, mistralai/Mistral-7B-v0.1
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
    if checkpoint.startswith("text-da"):
        return LLM_OpenAI(checkpoint, seed=seed, CACHE_DIR=CACHE_DIR)
    elif checkpoint.startswith("gpt-3") or checkpoint.startswith("gpt-4"):
        return LLM_Chat(checkpoint, seed, role, CACHE_DIR)
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


class LLM_OpenAI:
    def __init__(self, checkpoint, seed, CACHE_DIR):
        self.cache_dir = join(
            CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
        )
        self.checkpoint = checkpoint

    @repeatedly_call_with_delay
    def __call__(
        self,
        prompt: str,
        max_new_tokens=250,
        do_sample=True,
        stop=None,
        return_str=True,
    ):
        import openai

        # cache
        os.makedirs(self.cache_dir, exist_ok=True)
        hash_str = hashlib.sha256(prompt.encode()).hexdigest()
        cache_file = join(
            self.cache_dir, f"{hash_str}__num_tok={max_new_tokens}.pkl")
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
            stop=stop,
            # stop=["101"]
        )
        if return_str:
            response = response["choices"][0]["text"]

        pkl.dump(response, open(cache_file, "wb"))
        return response


class LLM_Chat:
    """Chat models take a different format: https://platform.openai.com/docs/guides/chat/introduction"""

    def __init__(self, checkpoint, seed, role, CACHE_DIR):
        self.cache_dir = join(
            CACHE_DIR, "cache_openai", f'{checkpoint.replace("/", "_")}___{seed}'
        )
        self.checkpoint = checkpoint
        self.role = role
        from openai import AzureOpenAI
        self.client = AzureOpenAI(
            api_version="2023-07-01-preview",
            azure_endpoint=open(os.path.expanduser(
                '~/.AZURE_OPENAI_ENDPOINT')).read().strip(),
            api_key=open(os.path.expanduser(
                '~/.AZURE_OPENAI_KEY')).read().strip(),
        )

    @repeatedly_call_with_delay
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
    ):
        """
        prompts_list: list of dicts, each dict has keys 'role' and 'content'
            Example: [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
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
        if os.path.exists(cache_file):
            if verbose:
                print("cached!")
                # print(cache_file)
            # print(cache_file)
            return pkl.load(open(cache_file, "rb"))
        if verbose:
            print("not cached")

        kwargs = dict(
            model=self.checkpoint,
            messages=prompts_list,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=frequency_penalty,  # maximum is 2
            presence_penalty=0,
            stop=stop,
            # stop=["101"]
        )
        if functions is not None:
            kwargs["functions"] = functions

        completion = self.client.chat.completions.create(
            **kwargs,
        )

        if return_str:
            response = completion.choices[0].message.content

        pkl.dump(response, open(cache_file, "wb"))

        return response


def load_tokenizer(checkpoint: str) -> transformers.PreTrainedTokenizer:
    if "facebook/opt" in checkpoint:
        # opt can't use fast tokenizer
        return AutoTokenizer.from_pretrained(checkpoint, use_fast=False, padding_side='left')
    elif "PMC_LLAMA" in checkpoint:
        return transformers.LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B", padding_side='left')
    else:
        # , use_fast=True)
        return AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
    # return AutoTokenizer.from_pretrained(checkpoint,
    # token=os.environ.get("LLAMA_TOKEN"),)


class LLM_HF:
    def __init__(self, checkpoint, seed, CACHE_DIR, LLAMA_DIR=None):
        self._tokenizer = load_tokenizer(checkpoint)

        # set checkpoint
        kwargs = {
            "pretrained_model_name_or_path": checkpoint,
            "output_hidden_states": False,
            # "pad_token_id": tokenizer.eos_token_id,
            "low_cpu_mem_usage": True,
        }
        if "google/flan" in checkpoint:
            self._model = T5ForConditionalGeneration.from_pretrained(
                checkpoint, device_map="auto", torch_dtype=torch.float16
            )
        elif checkpoint == "EleutherAI/gpt-j-6B":
            self._model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                revision="float16",
                torch_dtype=torch.float16,
                **kwargs,
            )
        elif "llama-2" in checkpoint.lower():
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=torch.float16,
                device_map="auto",
                token=os.environ.get("LLAMA_TOKEN"),
                offload_folder="offload",
            )
        elif "llama_" in checkpoint:
            self._model = transformers.LlamaForCausalLM.from_pretrained(
                join(LLAMA_DIR, checkpoint),
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif 'microsoft/phi' in checkpoint:
            self._model = AutoModelForCausalLM.from_pretrained(
                checkpoint
            )
        elif checkpoint == "gpt-xl":
            self._model = AutoModelForCausalLM.from_pretrained(checkpoint)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                checkpoint, device_map="auto", torch_dtype=torch.float16
            )
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
        # batch_size=1,
    ) -> Union[str, List[str]]:
        """Warning: stop is used posthoc but not during generation.
        Be careful, caching can take up a lot of memory....


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
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            inputs = self._tokenizer(
                prompt, return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                truncation=False,
            ).to(self._model.device)

            # torch.manual_seed(0)
            if return_next_token_prob_scores:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=1,
                    pad_token_id=self._tokenizer.pad_token_id,
                    output_logits=True,
                    return_dict_in_generate=True,
                )
                next_token_logits = outputs['logits'][0]
                next_token_probs = next_token_logits.softmax(
                    axis=-1).detach().cpu().numpy()

                if target_token_strs is not None:
                    target_token_ids = self._check_target_token_strs(
                        target_token_strs)
                    if return_top_target_token_str:
                        selected_tokens = next_token_probs[:, np.array(
                            target_token_ids)].squeeze().argmax(axis=-1)
                        out_strs = [
                            target_token_strs[selected_tokens[i]]
                            for i in range(len(selected_tokens))
                        ]
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
                    return next_token_probs
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
                # top_p=0.92,
                # temperature=0,
                # top_k=0
            if input_is_str:
                out_str = self._tokenizer.decode(
                    outputs[0], skip_special_tokens=True)
                out_str = out_str[len(prompt):]
                if use_cache:
                    pkl.dump(out_str, open(cache_file, "wb"))
                return out_str
            else:
                out_strs = []
                for i in range(outputs.shape[0]):
                    out_tokens = outputs[i]
                    out_str = self._tokenizer.decode(
                        out_tokens, skip_special_tokens=True)
                    out_str = out_str[len(prompt[i]):]
                    out_strs.append(out_str)
                if use_cache:
                    pkl.dump(out_strs, open(cache_file, "wb"))
                return out_strs

    def _check_target_token_strs(self, target_token_strs, override_token_with_first_token_id=False):
        # deal with target_token_strs.... ######################
        if isinstance(target_token_strs, str):
            target_token_strs = [target_token_strs]

        target_token_ids = [self._tokenizer(target_token_str)["input_ids"]
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
                        str([self._tokenizer.decode(target_token_id)
                            for target_token_id in target_token_ids[i]]))
        return target_token_ids


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

    # FORCE WORDSSSSSSSSS ##########
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
