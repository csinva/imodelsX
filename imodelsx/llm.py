'''Lightweight wrapper for calling LLMs from different providers, with caching.

Example usage:
    import imodelsx.llm
    llm = imodelsx.llm.get_llm('gpt2')                        # local huggingface model
    llm('may the force be')                                   # returns ' with you'
    llm = imodelsx.llm.get_llm('claude-code:haiku')           # claude code CLI (uses your claude login)
    llm('What is the capital of Spain?')                      # returns 'Madrid'

Checkpoints can be prefixed with a provider as "provider:model" (see PROVIDERS).
Without a prefix the provider is inferred from the checkpoint name:
    gpt-*, o1*, o3*, o4*  -> openai (if an OpenAI key is found, else the legacy azure setup)
    claude-*              -> anthropic (if an Anthropic key is found, else claude-code)
    anything else         -> hf (local huggingface model)

API keys are looked up in this order: the api_key argument, the environment variable
in API_KEY_ENV_VARS, then a file at ~/.<ENV_VAR_NAME> (e.g. ~/.OPENAI_API_KEY).
'''
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import os
from os.path import join, expanduser
import pickle as pkl
import hashlib
import shutil
import subprocess
import tempfile
import torch
import traceback
import time
from tqdm import tqdm

# change these settings before using these classes!
LLM_CONFIG = {
    # how long to wait before recalling a failed llm call (can set to None)
    "LLM_REPEAT_DELAY": 10,
    "CACHE_DIR": join(
        os.path.expanduser("~"), ".CACHE_LLM"
    ),  # path to save cached llm outputs
}

PROVIDERS = {
    'hf': 'local huggingface transformers model',
    'hf-api': 'huggingface inference providers API (huggingface_hub.InferenceClient)',
    'openai': 'OpenAI API',
    'azure': 'Azure OpenAI',
    'anthropic': 'Anthropic API',
    'claude-code': 'Claude Code CLI (`claude -p`), uses your Claude Code login',
    'codex': 'OpenAI Codex CLI (`codex exec`), uses your Codex login',
}
API_KEY_ENV_VARS = {
    'hf': 'HF_TOKEN',
    'hf-api': 'HF_TOKEN',
    'openai': 'OPENAI_API_KEY',
    'azure': 'AZURE_OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
}
OPENAI_PREFIXES = ("gpt-3", "gpt-4", "gpt-5", "o1", "o3", "o4")


def get_api_key(provider: str, api_key: Optional[str] = None) -> Optional[str]:
    """Return the api key for a provider (argument > env var > ~/.<ENV_VAR> file)"""
    if api_key:
        return api_key
    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var is None:
        return None
    if os.environ.get(env_var):
        return os.environ[env_var]
    key_file = expanduser(f'~/.{env_var}')
    if os.path.exists(key_file):
        return open(key_file, 'r').read().strip()
    return None


# keep the old behavior of reading ~/.HF_TOKEN into the environment
if not os.environ.get('HF_TOKEN') and get_api_key('hf'):
    os.environ['HF_TOKEN'] = get_api_key('hf')


def parse_checkpoint(checkpoint: str, provider: Optional[str] = None) -> Tuple[str, str]:
    """Split a checkpoint into (provider, model), inferring the provider if needed"""
    if provider is None and ':' in checkpoint:
        prefix, model = checkpoint.split(':', 1)
        if prefix in PROVIDERS:
            provider, checkpoint = prefix, model
    if provider is None:
        if checkpoint.startswith(OPENAI_PREFIXES):
            provider = 'openai' if get_api_key('openai') else 'azure'
        elif checkpoint.startswith('claude'):
            provider = 'anthropic' if get_api_key('anthropic') else 'claude-code'
        else:
            provider = 'hf'
    if provider not in PROVIDERS:
        raise ValueError(
            f'Unknown provider {provider}, must be one of {list(PROVIDERS)}')
    return provider, checkpoint


def is_api_checkpoint(checkpoint: str) -> bool:
    """True if the checkpoint is served by an API / CLI provider (chat-style) rather than a local HF model"""
    return parse_checkpoint(checkpoint)[0] != 'hf'


def get_llm(
    checkpoint,
    seed=1,
    role: str = None,
    repeat_delay: Optional[float] = None,
    CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    """Get an LLM with a call function and caching capabilities

    Params
    ------
    checkpoint: str
        Model name, optionally prefixed with a provider, e.g. 'gpt2', 'Qwen/Qwen2.5-0.5B-Instruct',
        'gpt-4o', 'anthropic:claude-haiku-5-5', 'claude-code:haiku', 'codex:gpt-5-codex', 'hf-api:meta-llama/Llama-3.1-8B-Instruct'
    seed: int
        Seed (part of the cache key; for hf models this also sets torch.manual_seed)
    role: str
        System prompt for chat models
    repeat_delay: float
        Seconds to wait before retrying a failed API call (None raises the error instead)
    CACHE_DIR: str
        Where to cache outputs (None disables caching)
    provider: str
        One of PROVIDERS (overrides any prefix / inference from the checkpoint)
    api_key: str
        API key / token for the provider (otherwise read from env var or ~/.<ENV_VAR>)
    kwargs
        Passed to the LLM class (e.g. cli_path, timeout for claude-code / codex)
    """
    if repeat_delay is not None:
        LLM_CONFIG["LLM_REPEAT_DELAY"] = repeat_delay

    provider, checkpoint = parse_checkpoint(checkpoint, provider)
    if provider == 'hf':
        token = get_api_key('hf', api_key)
        if 'meta-llama' in checkpoint and 'Instruct' in checkpoint:
            if token is None:
                raise ValueError(
                    "You must set the HF_TOKEN environment variable (or pass api_key) to use this model.")
            return LLM_HF_Pipeline(checkpoint, CACHE_DIR, token=token)
        # warning: this sets torch.manual_seed(seed)
        return LLM_HF(checkpoint, seed=seed, CACHE_DIR=CACHE_DIR, token=token)

    llm_class = {
        'openai': LLM_Chat,
        'azure': LLM_Chat,
        'anthropic': LLM_Anthropic,
        'claude-code': LLM_ClaudeCode,
        'codex': LLM_Codex,
        'hf-api': LLM_HF_API,
    }[provider]
    return llm_class(checkpoint, seed=seed, role=role, CACHE_DIR=CACHE_DIR,
                     api_key=api_key, provider=provider, **kwargs)


# errors that will not go away by retrying
NON_RETRYABLE_ERRORS = (ImportError, FileNotFoundError,
                        NotImplementedError, TypeError)


def repeatedly_call_with_delay(llm_call):
    def wrapper(*args, **kwargs):
        # Number of seconds to wait between calls (None will not repeat)
        delay = LLM_CONFIG["LLM_REPEAT_DELAY"]
        response = None
        while response is None:
            try:
                response = llm_call(*args, **kwargs)
            except NON_RETRYABLE_ERRORS:
                raise
            except Exception as e:
                e_str = str(e)
                print(e_str)
                if "does not exist" in e_str:
                    return None
                elif "maximum context length" in e_str:
                    return None
                elif 'content management policy' in e_str:
                    return None
                if delay is None:
                    raise e
                else:
                    time.sleep(delay)
        return response

    return wrapper


def _flatten_messages(messages: List[Dict[str, str]]) -> Tuple[str, str]:
    """Convert chat messages into (system prompt, single prompt string) for single-prompt CLIs"""
    system = '\n\n'.join(m['content']
                         for m in messages if m['role'] == 'system')
    turns = [m for m in messages if m['role'] != 'system']
    if len(turns) == 1 and turns[0]['role'] == 'user':
        return system, turns[0]['content']
    return system, '\n\n'.join(f"{m['role'].capitalize()}: {m['content']}" for m in turns)


def _truncate_at_stop(text: str, stop) -> str:
    if text is None or stop is None:
        return text
    for s in ([stop] if isinstance(stop, str) else stop):
        if s in text:
            text = text[:text.index(s)]
    return text


class LLM_ChatBase:
    """Shared logic for chat-style API / CLI models: prompt formatting, caching, retries, and batching.
    Subclasses implement _generate.
    """
    provider = None
    # whether the provider applies stop sequences itself (otherwise they are applied posthoc)
    supports_stop = True

    def __init__(self, checkpoint, seed=1, role=None, CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
                 api_key=None, provider=None, num_threads=8):
        if provider is not None:
            self.provider = provider
        cache_subdir = 'cache_openai' if self.provider in ['openai', 'azure'] \
            else f'cache_{self.provider}'
        self.cache_dir = join(
            CACHE_DIR, cache_subdir, f'{checkpoint.replace("/", "_")}___{seed}'
        ) if CACHE_DIR is not None else None
        self.checkpoint = checkpoint
        self.role = role
        self.api_key = get_api_key(self.provider, api_key)
        self.num_threads = num_threads
        self.reasoning_effort = None
        self.reasoning_effort_fixed = False

    def __call__(
        self,
        prompts_list: Union[str, List[str], List[Dict[str, str]]],
        max_completion_tokens=250,
        stop=None,
        functions: List[Dict] = None,
        return_str=True,
        verbose=True,
        temperature=0,
        frequency_penalty=0.25,
        use_cache=True,
        return_false_if_not_cached=False,
        reasoning_effort=None,
        seed=1,
        max_new_tokens=None,
        do_sample=False,
        batch_size=None,
        num_threads=None,
    ):
        """
        prompts_list: str
            A prompt, which gets formatted into basic messages:
            [
                {"role": "system", "content": role or "You are a helpful assistant."},
                {"role": "user", "content": <<<<<prompts_list>>>>},
            ]
        prompts_list: List[Dict[str, str]]
            A list of chat messages, each dict has keys 'role' and 'content'
        prompts_list: List[str]
            A batch of prompts, called in parallel with num_threads threads (returns a list)
        max_new_tokens, do_sample, batch_size
            Accepted for compatibility with LLM_HF (max_new_tokens overrides max_completion_tokens,
            do_sample sets temperature=1 if temperature is 0, batch_size is ignored)
        """
        if max_new_tokens is not None:
            max_completion_tokens = max_new_tokens
        if do_sample and temperature == 0:
            temperature = 1
        kwargs = dict(
            max_completion_tokens=max_completion_tokens, stop=stop, functions=functions,
            return_str=return_str, verbose=verbose, temperature=temperature,
            frequency_penalty=frequency_penalty, use_cache=use_cache,
            return_false_if_not_cached=return_false_if_not_cached,
            reasoning_effort=reasoning_effort, seed=seed,
        )
        if isinstance(prompts_list, (list, tuple, np.ndarray)) and len(prompts_list) > 0 \
                and all(isinstance(p, str) for p in prompts_list):
            with ThreadPoolExecutor(num_threads or self.num_threads) as executor:
                return list(executor.map(lambda p: self._call_single(p, **kwargs), prompts_list))
        return self._call_single(prompts_list, **kwargs)

    @repeatedly_call_with_delay
    def _call_single(self, prompts_list, max_completion_tokens, stop, functions, return_str,
                     verbose, temperature, frequency_penalty, use_cache,
                     return_false_if_not_cached, reasoning_effort, seed):
        if isinstance(prompts_list, str):
            role = self.role
            if role is None:
                role = "You are a helpful assistant."
            prompts_list = [
                {"role": "system", "content": role},
                {"role": "user", "content": prompts_list},
            ]
        assert isinstance(prompts_list, list), prompts_list

        if self.reasoning_effort_fixed:
            if reasoning_effort not in [None, 'high', self.reasoning_effort]:
                raise ValueError(
                    f"reasoning_effort is fixed to {self.reasoning_effort} for this model!")
            reasoning_effort = self.reasoning_effort

        # cache (key is kept identical to earlier versions so old caches stay valid)
        use_cache = use_cache and self.cache_dir is not None
        cache_file = None
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            prompts_list_dict = {
                str(i): sorted(v.items()) for i, v in enumerate(prompts_list)
            }
            prompts_list_dict["checkpoint"] = self.checkpoint
            prompts_list_dict["temperature"] = temperature
            prompts_list_dict["functions"] = functions
            prompts_list_dict["max_completion_tokens"] = max_completion_tokens
            if not seed == 1:
                prompts_list_dict["seed"] = seed
            if reasoning_effort not in [None, 'high']:
                prompts_list_dict['reasoning_effort'] = reasoning_effort
            if stop is not None and not self.supports_stop:
                prompts_list_dict['stop'] = stop
            dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
            hash_str = hashlib.sha256(dict_as_str.encode()).hexdigest()
            cache_file = join(self.cache_dir, f"chat__{hash_str}.pkl")
            if os.path.exists(cache_file):
                if verbose:
                    print("cached!")
                response = pkl.load(open(cache_file, "rb"))
                if response is not None:
                    return response
            if verbose:
                print("not cached")

        if return_false_if_not_cached:
            return False

        response = self._generate(
            prompts_list,
            max_completion_tokens=max_completion_tokens,
            stop=stop,
            functions=functions,
            return_str=return_str,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            reasoning_effort=reasoning_effort,
        )
        if return_str and not self.supports_stop:
            response = _truncate_at_stop(response, stop)

        if response is not None and use_cache:
            try:
                pkl.dump(response, open(cache_file, "wb"))
            except:
                print('failed to save cache!', cache_file)
                traceback.print_exc()
        return response

    def _generate(self, messages, max_completion_tokens, stop, functions, return_str,
                  temperature, frequency_penalty, reasoning_effort):
        raise NotImplementedError


class LLM_Chat(LLM_ChatBase):
    """OpenAI / Azure OpenAI chat models: https://platform.openai.com/docs/guides/chat/introduction

    provider='openai' uses the OpenAI API with OPENAI_API_KEY (OPENAI_BASE_URL is respected,
    so any OpenAI-compatible server works too).
    provider='azure' uses AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY if set,
    otherwise falls back to Azure CLI / managed identity credentials.
    """
    provider = 'openai'

    def __init__(self, checkpoint, seed=1, role=None, CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
                 api_key=None, provider=None, **kwargs):
        if provider is None:
            provider = 'openai' if get_api_key('openai', api_key) else 'azure'
        super().__init__(checkpoint, seed, role, CACHE_DIR, api_key, provider, **kwargs)

        # gpt-5 checkpoints can fix the reasoning effort with a suffix, e.g. gpt-5-mini-low
        if 'gpt-5' in checkpoint:
            for suffix in ['-low', '-medium', '-high', '-none', '-minimal']:
                if self.checkpoint.endswith(suffix):
                    self.checkpoint = self.checkpoint[:-len(suffix)]
                    self.reasoning_effort = suffix[1:]
                    self.reasoning_effort_fixed = True

        if self.provider == 'openai':
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        elif os.environ.get('AZURE_OPENAI_ENDPOINT') and self.api_key:
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                api_version=os.environ.get(
                    'OPENAI_API_VERSION', '2025-01-01-preview'),
            )
        else:
            self._init_azure_identity_client(checkpoint)

    def _init_azure_identity_client(self, checkpoint):
        """Legacy setup: Azure endpoints authenticated with Azure CLI / managed identity"""
        try:
            from openai import AzureOpenAI
            from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
            client_id = os.environ.get("AZURE_CLIENT_ID")
            scope = "https://cognitiveservices.azure.com/.default"
            credential = get_bearer_token_provider(ChainedTokenCredential(
                AzureCliCredential(),  # first check local
                ManagedIdentityCredential(client_id=client_id)
            ), scope)
            if 'audio' in checkpoint:
                self.client = AzureOpenAI(
                    api_version="2025-01-01-preview",
                    azure_endpoint="https://neuroaiservice.cognitiveservices.azure.com/openai/deployments/gpt-4o-audio-preview/chat/completions?api-version=2025-01-01-preview",
                    azure_ad_token_provider=credential,
                    timeout=10,
                    max_retries=3,
                )
            elif 'gpt-5' in checkpoint:
                self.client = AzureOpenAI(
                    api_version="2025-01-01-preview",
                    azure_endpoint="https://dl-openai-3.openai.azure.com/",
                    azure_ad_token_provider=credential
                )
            else:
                self.client = AzureOpenAI(
                    api_version="2025-01-01-preview",
                    azure_endpoint="https://dl-openai-1.openai.azure.com/",
                    azure_ad_token_provider=credential
                )
        except Exception as e:
            print('failed to create client', e)
            print('Set OPENAI_API_KEY (or AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY) to use OpenAI models.')
            traceback.print_exc()

    def _generate(self, messages, max_completion_tokens, stop, functions, return_str,
                  temperature, frequency_penalty, reasoning_effort):
        kwargs = dict(
            model=self.checkpoint,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=frequency_penalty,  # maximum is 2
            presence_penalty=0,
            stop=stop,
        )
        if functions is not None:
            kwargs["functions"] = functions
        if self.checkpoint.startswith('o') or self.checkpoint.startswith('gpt-5'):
            # reasoning models don't support sampling params
            del kwargs['temperature']
            del kwargs['frequency_penalty']
            del kwargs['top_p']
        if 'gpt-5' in self.checkpoint:
            kwargs['reasoning_effort'] = reasoning_effort or 'high'

        response = self.client.chat.completions.create(**kwargs)
        if return_str:
            response = response.choices[0].message.content
        return response


class LLM_Anthropic(LLM_ChatBase):
    """Claude models through the Anthropic API (pip install anthropic).
    The key comes from api_key / ANTHROPIC_API_KEY / ~/.ANTHROPIC_API_KEY
    (the SDK also picks up ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile).

    Note: current Claude models reject non-default sampling params, so temperature and
    frequency_penalty are not sent. They also think before answering, so max_tokens is
    raised to at least MIN_MAX_TOKENS to leave room for the answer.
    """
    provider = 'anthropic'
    MIN_MAX_TOKENS = 1024

    def __init__(self, checkpoint, seed=1, role=None, CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
                 api_key=None, provider=None, **kwargs):
        super().__init__(checkpoint, seed, role, CACHE_DIR, api_key, provider, **kwargs)
        import anthropic
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key \
            else anthropic.Anthropic()

    def _generate(self, messages, max_completion_tokens, stop, functions, return_str,
                  temperature, frequency_penalty, reasoning_effort):
        system = '\n\n'.join(m['content']
                             for m in messages if m['role'] == 'system')
        kwargs = dict(
            model=self.checkpoint,
            max_tokens=max(max_completion_tokens, self.MIN_MAX_TOKENS),
            messages=[m for m in messages if m['role'] != 'system'],
        )
        if system:
            kwargs['system'] = system
        if stop is not None:
            kwargs['stop_sequences'] = [stop] if isinstance(stop, str) else list(stop)
        if reasoning_effort is not None:
            kwargs['output_config'] = {'effort': reasoning_effort}
        response = self.client.messages.create(**kwargs)
        if return_str:
            return ''.join(b.text for b in response.content if b.type == 'text')
        return response


# env vars that mark a process as running inside a Claude Code session
# (removed so that the `claude` subprocess starts a fresh, independent session)
_CLAUDE_SESSION_ENV_VARS = [
    'CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT', 'CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION',
    'CLAUDE_CODE_BRIDGE_SESSION_ID', 'CLAUDE_CODE_MESSAGING_SOCKET', 'CLAUDE_CODE_MESSAGING_TOKEN',
    'CLAUDE_CODE_AGENT', 'CLAUDE_CODE_SESSION_ATTENDED', 'CLAUDE_PID',
]


class LLM_ClaudeCode(LLM_ChatBase):
    """Claude through the Claude Code CLI (`claude -p`), using your existing Claude Code login
    (no API key needed). Install: https://code.claude.com, then run `claude` once to log in.

    checkpoint is passed to `claude --model`, e.g. 'haiku', 'sonnet', 'opus', or a full id like 'claude-haiku-5-5'.
    The CLI runs with all tools, settings, and MCP servers disabled, so it acts as a plain text model.
    It has no max-token, temperature, or stop controls (stop is applied posthoc).
    """
    provider = 'claude-code'
    supports_stop = False

    def __init__(self, checkpoint, seed=1, role=None, CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
                 api_key=None, provider=None, cli_path='claude', timeout=600, **kwargs):
        super().__init__(checkpoint, seed, role, CACHE_DIR, api_key, provider, **kwargs)
        self.cli_path = shutil.which(cli_path) or cli_path
        self.timeout = timeout

    def _generate(self, messages, max_completion_tokens, stop, functions, return_str,
                  temperature, frequency_penalty, reasoning_effort):
        system, prompt = _flatten_messages(messages)
        cmd = [
            self.cli_path, '-p', '--model', self.checkpoint, '--output-format', 'json',
            '--tools', '', '--setting-sources', '', '--strict-mcp-config', '--no-session-persistence',
        ]
        if system:
            cmd += ['--system-prompt', system]
        if reasoning_effort is not None:
            cmd += ['--effort', reasoning_effort]
        env = {k: v for k, v in os.environ.items()
               if k not in _CLAUDE_SESSION_ENV_VARS}
        if self.api_key:
            env['ANTHROPIC_API_KEY'] = self.api_key
        proc = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            cwd=tempfile.gettempdir(), env=env, timeout=self.timeout,
        )
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise RuntimeError(
                f'claude CLI failed (exit {proc.returncode}): {proc.stderr or proc.stdout}')
        if out.get('is_error') or proc.returncode != 0:
            raise RuntimeError(f'claude CLI error: {out.get("result", out)}')
        return out['result'] if return_str else out


class LLM_Codex(LLM_ChatBase):
    """OpenAI models through the Codex CLI (`codex exec`), using your existing Codex login
    (ChatGPT account or OPENAI_API_KEY). Install: `npm i -g @openai/codex`, then `codex login`.

    checkpoint is passed to `codex exec --model` (use 'default' for the CLI's configured model).
    The CLI runs in a read-only sandbox in an empty temp dir; it has no max-token,
    temperature, or stop controls (stop is applied posthoc).
    """
    provider = 'codex'
    supports_stop = False

    def __init__(self, checkpoint, seed=1, role=None, CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
                 api_key=None, provider=None, cli_path='codex', timeout=600, **kwargs):
        super().__init__(checkpoint, seed, role, CACHE_DIR, api_key, provider, **kwargs)
        self.cli_path = shutil.which(cli_path) or cli_path
        self.timeout = timeout

    def _generate(self, messages, max_completion_tokens, stop, functions, return_str,
                  temperature, frequency_penalty, reasoning_effort):
        system, prompt = _flatten_messages(messages)
        if system:
            prompt = f'{system}\n\n{prompt}'
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = join(tmp_dir, 'last_message.txt')
            cmd = [self.cli_path, 'exec', '--skip-git-repo-check', '--sandbox', 'read-only',
                   '--cd', tmp_dir, '--output-last-message', out_file]
            if self.checkpoint and self.checkpoint != 'default':
                cmd += ['--model', self.checkpoint]
            if reasoning_effort is not None:
                cmd += ['-c', f'model_reasoning_effort="{reasoning_effort}"']
            cmd.append('-')  # read the prompt from stdin
            env = dict(os.environ)
            openai_key = get_api_key('openai', self.api_key)
            if openai_key:
                env['OPENAI_API_KEY'] = openai_key
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                cwd=tmp_dir, env=env, timeout=self.timeout,
            )
            if proc.returncode != 0 or not os.path.exists(out_file):
                raise RuntimeError(
                    f'codex CLI failed (exit {proc.returncode}): {proc.stderr[-2000:]}')
            return open(out_file).read().strip()


class LLM_HF_API(LLM_ChatBase):
    """Chat models served through huggingface inference providers (huggingface_hub.InferenceClient),
    authenticated with HF_TOKEN. checkpoint is a hub model id, e.g. 'meta-llama/Llama-3.1-8B-Instruct'
    (append ':<provider>' to the model id to pick a specific inference provider).
    """
    provider = 'hf-api'

    def __init__(self, checkpoint, seed=1, role=None, CACHE_DIR=LLM_CONFIG["CACHE_DIR"],
                 api_key=None, provider=None, **kwargs):
        super().__init__(checkpoint, seed, role, CACHE_DIR, api_key, provider, **kwargs)
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(token=self.api_key)

    def _generate(self, messages, max_completion_tokens, stop, functions, return_str,
                  temperature, frequency_penalty, reasoning_effort):
        kwargs = dict(model=self.checkpoint, messages=messages,
                      max_tokens=max_completion_tokens)
        if temperature:
            kwargs['temperature'] = temperature
        if stop is not None:
            kwargs['stop'] = [stop] if isinstance(stop, str) else list(stop)
        response = self.client.chat_completion(**kwargs)
        if return_str:
            return response.choices[0].message.content
        return response


def load_tokenizer(checkpoint: str, token: Optional[str] = None) -> transformers.PreTrainedTokenizer:
    token = token or os.environ.get('HF_TOKEN')
    if "facebook/opt" in checkpoint:
        # opt can't use fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, use_fast=False, padding_side='left', token=token)
    elif "PMC_LLAMA" in checkpoint:
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "chaoyi-wu/PMC_LLAMA_7B", padding_side='left', token=token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, padding_side='left', use_fast=True, token=token)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_hf_model(checkpoint: str, token: Optional[str] = None) -> transformers.PreTrainedModel:
    token = token or os.environ.get('HF_TOKEN')
    # float16 is slow / unsupported for many ops on cpu
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if "google/flan" in checkpoint:
        return T5ForConditionalGeneration.from_pretrained(
            checkpoint, device_map="auto", torch_dtype=dtype
        )
    elif checkpoint == "EleutherAI/gpt-j-6B":
        return AutoModelForCausalLM.from_pretrained(
            checkpoint,
            revision="float16",
            torch_dtype=dtype,
            output_hidden_states=False,
            low_cpu_mem_usage=True,
        )
    elif "llama-2" in checkpoint.lower():
        return AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            device_map="auto",
            token=token,
            offload_folder="offload",
        )
    elif 'microsoft/phi' in checkpoint or checkpoint == "gpt-xl":
        return AutoModelForCausalLM.from_pretrained(checkpoint)
    else:
        return AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            torch_dtype=dtype,
            token=token,
        )


class LLM_HF_Pipeline:
    def __init__(self, checkpoint, CACHE_DIR, token=None):
        self.pipeline_ = transformers.pipeline(
            "text-generation",
            model=checkpoint,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            token=token or os.environ.get('HF_TOKEN'),
        )
        self.pipeline_.tokenizer.pad_token_id = self.pipeline_.tokenizer.eos_token_id
        self.pipeline_.tokenizer.padding_side = 'left'
        self.cache_dir = CACHE_DIR

    def __call__(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens=20,
        use_cache=True,
        verbose=False,
        batch_size=64,
    ):
        use_cache = use_cache and self.cache_dir is not None
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
            top_p=None,
            temperature=None,
            return_full_text=False,
        )
        if isinstance(prompt, str):
            texts = outputs[0]["generated_text"]
        else:
            texts = [outputs[i][0]['generated_text']
                     for i in range(len(outputs))]

        if use_cache:
            pkl.dump(texts, open(cache_file, "wb"))
        return texts


class LLM_Chat_Audio(LLM_Chat):

    def __call__(
        self,
        prompt_str: str,
        audio_str: str,
        return_str=True,
        verbose=True,
        use_cache=True,
        return_false_if_not_cached=False,
    ):
        prompts_list = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_str,
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_str,
                        "format": "wav"
                    }
                }
            ]
        }]

        # cache
        os.makedirs(self.cache_dir, exist_ok=True)
        prompts_list_dict = {
            str(i): sorted(v.items()) for i, v in enumerate(prompts_list)
        }
        dict_as_str = json.dumps(prompts_list_dict, sort_keys=True)
        hash_str = hashlib.sha256(dict_as_str.encode()).hexdigest()
        cache_file = join(
            self.cache_dir,
            f"audio__{hash_str}.pkl",
        )
        if os.path.exists(cache_file) and use_cache:
            if verbose:
                print("cached!")
            response = pkl.load(open(cache_file, "rb"))
            if response is not None:
                return response
        if verbose:
            print("not cached")

        if return_false_if_not_cached:
            return False

        response = self.client.chat.completions.create(
            model=self.checkpoint,
            messages=prompts_list,
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
        )

        if return_str:
            response = response.choices[0].message.audio.transcript

        if response is not None:
            try:
                pkl.dump(response, open(cache_file, "wb"))
            except:
                print('failed to save cache!', cache_file)
                traceback.print_exc()

        return response


class LLM_HF:
    def __init__(self, checkpoint, seed, CACHE_DIR, token=None):
        self.tokenizer_ = load_tokenizer(checkpoint, token=token)
        self.model_ = load_hf_model(checkpoint, token=token)
        self.checkpoint = checkpoint
        if CACHE_DIR is not None:
            self.cache_dir = join(
                CACHE_DIR, "cache_hf", f'{checkpoint.replace("/", "_")}___{seed}'
            )
        else:
            self.cache_dir = None
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
        """Warning: stop is applied posthoc, not during generation.
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
            use_cache = use_cache and self.cache_dir is not None
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

            outputs = self.model_.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self.tokenizer_.pad_token_id,
            )
            # decoder-only models return prompt + generation (prompts are left-padded to the same length)
            if not getattr(self.model_.config, 'is_encoder_decoder', False):
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
            out_strs = [
                _truncate_at_stop(self.tokenizer_.decode(
                    out_tokens, skip_special_tokens=True), stop)
                for out_tokens in outputs
            ]
            if input_is_str:
                out_strs = out_strs[0]
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
    def __init__(self, checkpoint, token=None):
        self.tokenizer_ = load_tokenizer(checkpoint, token=token)
        self.model_ = AutoModel.from_pretrained(
            checkpoint, output_hidden_states=True,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=token or os.environ.get('HF_TOKEN'),
        )

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
    llm = get_llm("claude-code:haiku")
    print(llm("What is the capital of spain? Answer in one word.", use_cache=False))

    llm = get_llm("gpt2")
    print(llm("may the force be with", use_cache=False))

    # next-token probabilities for a fixed set of target tokens
    print(llm(['roses are red, violets are', 'may the force be with'], use_cache=False,
              return_next_token_prob_scores=True, target_token_strs=[' blue', ' you']))
