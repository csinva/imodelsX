'''Offline tests for provider routing, keys, caching, and the API / CLI wrappers in imodelsx.llm.
Network calls and CLIs are replaced with fakes (see test_live.py for real calls).'''
import json
import sys
from types import SimpleNamespace

import pytest

import imodelsx.llm as llm_mod


@pytest.fixture(autouse=True)
def no_keys(monkeypatch, tmp_path):
    for env_var in set(llm_mod.API_KEY_ENV_VARS.values()):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv('HOME', str(tmp_path))  # no ~/.<ENV_VAR> key files
    monkeypatch.setitem(llm_mod.LLM_CONFIG, 'LLM_REPEAT_DELAY', None)


@pytest.mark.parametrize('checkpoint, provider, model', [
    ('gpt2', 'hf', 'gpt2'),
    ('Qwen/Qwen2.5-0.5B-Instruct', 'hf', 'Qwen/Qwen2.5-0.5B-Instruct'),
    ('openai/gpt-oss-20b', 'hf', 'openai/gpt-oss-20b'),  # hf repo, not the openai provider
    ('gpt-4o', 'azure', 'gpt-4o'),  # no OpenAI key -> legacy azure
    ('claude-haiku-5-5', 'claude-code', 'claude-haiku-5-5'),  # no Anthropic key -> claude code
    ('claude-code:haiku', 'claude-code', 'haiku'),
    ('codex:gpt-5-codex', 'codex', 'gpt-5-codex'),
    ('anthropic:claude-haiku-5-5', 'anthropic', 'claude-haiku-5-5'),
    ('hf-api:meta-llama/Llama-3.1-8B-Instruct', 'hf-api', 'meta-llama/Llama-3.1-8B-Instruct'),
    ('hf:gpt2', 'hf', 'gpt2'),
])
def test_parse_checkpoint(checkpoint, provider, model):
    assert llm_mod.parse_checkpoint(checkpoint) == (provider, model)


def test_parse_checkpoint_with_keys(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')
    assert llm_mod.parse_checkpoint('gpt-4o') == ('openai', 'gpt-4o')
    assert llm_mod.parse_checkpoint('claude-haiku-5-5') == ('anthropic', 'claude-haiku-5-5')
    assert llm_mod.parse_checkpoint('gpt2', provider='hf-api') == ('hf-api', 'gpt2')
    with pytest.raises(ValueError):
        llm_mod.parse_checkpoint('gpt2', provider='not-a-provider')


def test_get_api_key_order(monkeypatch, tmp_path):
    assert llm_mod.get_api_key('anthropic') is None
    (tmp_path / '.ANTHROPIC_API_KEY').write_text('from-file\n')
    assert llm_mod.get_api_key('anthropic') == 'from-file'
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'from-env')
    assert llm_mod.get_api_key('anthropic') == 'from-env'
    assert llm_mod.get_api_key('anthropic', 'from-arg') == 'from-arg'
    assert llm_mod.get_api_key('claude-code') is None


class FakeRun:
    '''Stands in for subprocess.run and records each command'''

    def __init__(self, stdout='', returncode=0, write_file=None):
        self.calls, self.stdout, self.returncode, self.write_file = [], stdout, returncode, write_file

    def __call__(self, cmd, input, **kwargs):
        self.calls.append(dict(cmd=cmd, input=input, **kwargs))
        if self.write_file is not None:
            out_file = cmd[cmd.index('--output-last-message') + 1]
            open(out_file, 'w').write(self.write_file)
        return SimpleNamespace(stdout=self.stdout, stderr='', returncode=self.returncode)


def test_claude_code(monkeypatch, tmp_path):
    fake = FakeRun(stdout=json.dumps({'result': 'Madrid. END extra', 'is_error': False}))
    monkeypatch.setattr(llm_mod.subprocess, 'run', fake)
    monkeypatch.setenv('CLAUDECODE', '1')
    llm = llm_mod.get_llm('claude-code:haiku', role='Be terse.', CACHE_DIR=str(tmp_path))
    assert isinstance(llm, llm_mod.LLM_ClaudeCode)

    assert llm('Capital of Spain?', stop=' END', verbose=False) == 'Madrid.'
    call = fake.calls[0]
    assert call['input'] == 'Capital of Spain?'
    assert call['cmd'][call['cmd'].index('--model') + 1] == 'haiku'
    assert call['cmd'][call['cmd'].index('--tools') + 1] == ''
    assert call['cmd'][call['cmd'].index('--system-prompt') + 1] == 'Be terse.'
    assert 'CLAUDECODE' not in call['env']

    # second call is served from the cache
    assert llm('Capital of Spain?', stop=' END', verbose=False) == 'Madrid.'
    assert len(fake.calls) == 1

    # errors are raised (retries are disabled in this test)
    monkeypatch.setattr(llm_mod.subprocess, 'run', FakeRun(
        stdout=json.dumps({'result': 'Not logged in', 'is_error': True}), returncode=1))
    with pytest.raises(RuntimeError, match='Not logged in'):
        llm('another prompt', use_cache=False, verbose=False)


def test_claude_code_batch_and_messages(monkeypatch):
    fake = FakeRun(stdout=json.dumps({'result': 'ok', 'is_error': False}))
    monkeypatch.setattr(llm_mod.subprocess, 'run', fake)
    llm = llm_mod.get_llm('claude-code:haiku', CACHE_DIR=None)
    assert llm(['a', 'b', 'c'], verbose=False) == ['ok', 'ok', 'ok']
    assert sorted(c['input'] for c in fake.calls) == ['a', 'b', 'c']

    llm([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'hello'},
         {'role': 'user', 'content': 'bye'}], verbose=False)
    assert fake.calls[-1]['input'] == 'User: hi\n\nAssistant: hello\n\nUser: bye'


def test_codex(monkeypatch):
    fake = FakeRun(write_file='Madrid\n')
    monkeypatch.setattr(llm_mod.subprocess, 'run', fake)
    llm = llm_mod.get_llm('codex:gpt-5-codex', CACHE_DIR=None)
    assert llm('Capital of Spain?', verbose=False, reasoning_effort='low') == 'Madrid'
    cmd = fake.calls[0]['cmd']
    assert cmd[:2] == [llm.cli_path, 'exec'] and cmd[-1] == '-'
    assert cmd[cmd.index('--model') + 1] == 'gpt-5-codex'
    assert 'model_reasoning_effort="low"' in cmd
    assert fake.calls[0]['input'].startswith('You are a helpful assistant.\n\nCapital of Spain?')


def test_anthropic(monkeypatch):
    created = []

    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.messages = SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            created.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type='thinking', thinking=''),
                                            SimpleNamespace(type='text', text='Madrid')])

    monkeypatch.setitem(sys.modules, 'anthropic', SimpleNamespace(Anthropic=FakeAnthropic))
    llm = llm_mod.get_llm('anthropic:claude-haiku-5-5', api_key='sk-ant-test', CACHE_DIR=None)
    assert llm.client.api_key == 'sk-ant-test'
    assert llm('Capital of Spain?', max_completion_tokens=1, stop='\n', verbose=False) == 'Madrid'
    kwargs = created[0]
    assert kwargs['model'] == 'claude-haiku-5-5'
    assert kwargs['system'] == 'You are a helpful assistant.'
    assert kwargs['messages'] == [{'role': 'user', 'content': 'Capital of Spain?'}]
    assert kwargs['max_tokens'] == llm_mod.LLM_Anthropic.MIN_MAX_TOKENS
    assert kwargs['stop_sequences'] == ['\n']
    assert 'temperature' not in kwargs


def test_openai(monkeypatch):
    created = []

    class FakeOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            created.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Madrid'))])

    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    llm = llm_mod.get_llm('gpt-4o-mini', CACHE_DIR=None)
    assert llm.provider == 'openai' and llm.client.api_key == 'sk-test'
    # max_new_tokens / do_sample are accepted for compatibility with hf models
    assert llm('Capital of Spain?', max_new_tokens=5, do_sample=True, verbose=False) == 'Madrid'
    assert created[0]['max_completion_tokens'] == 5 and created[0]['temperature'] == 1

    llm = llm_mod.get_llm('gpt-5-mini-low', CACHE_DIR=None)
    llm('hi', verbose=False)
    assert created[1]['model'] == 'gpt-5-mini' and created[1]['reasoning_effort'] == 'low'
    assert 'temperature' not in created[1]


def test_hf_api(monkeypatch):
    created = []

    class FakeInferenceClient:
        def __init__(self, token=None):
            self.token = token

        def chat_completion(self, **kwargs):
            created.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Madrid'))])

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, 'InferenceClient', FakeInferenceClient)
    monkeypatch.setenv('HF_TOKEN', 'hf_test')
    llm = llm_mod.get_llm('hf-api:meta-llama/Llama-3.1-8B-Instruct', CACHE_DIR=None)
    assert llm.client.token == 'hf_test'
    assert llm('Capital of Spain?', verbose=False) == 'Madrid'
    assert created[0]['model'] == 'meta-llama/Llama-3.1-8B-Instruct'
    assert 'temperature' not in created[0]


def test_missing_cli_is_not_retried(monkeypatch):
    monkeypatch.setitem(llm_mod.LLM_CONFIG, 'LLM_REPEAT_DELAY', 10)
    llm = llm_mod.get_llm('claude-code:haiku', CACHE_DIR=None,
                          cli_path='/nonexistent/claude')
    with pytest.raises(FileNotFoundError):
        llm('hi', verbose=False)
