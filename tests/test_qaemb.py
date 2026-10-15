from imodelsx import QAEmb
import imodelsx.llm
import numpy as np


def test_chat_token_limit(monkeypatch):
    calls = []

    # Match LLM_Chat's public call contract, without constructing credentials
    # or claiming that a live provider accepts the request.
    def chat(prompt, *, max_completion_tokens, verbose, use_cache):
        calls.append((prompt, max_completion_tokens, use_cache))
        return 'Yes'

    monkeypatch.setattr(imodelsx.llm, 'get_llm', lambda checkpoint, CACHE_DIR: chat)
    embedder = QAEmb(['Is this about film?'], checkpoint='gpt-4o', use_cache=False, CACHE_DIR=None)
    result = embedder(['a movie', 'a film'], verbose=False)
    np.testing.assert_array_equal(result, [[1.], [1.]])
    assert len(calls) == 2
    assert all(limit == 1 and cache is False for _, limit, cache in calls)
