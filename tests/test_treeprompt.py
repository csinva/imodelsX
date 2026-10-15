from types import SimpleNamespace
import numpy as np

from imodelsx import TreePromptClassifier
import imodelsx.llm
import imodelsx.treeprompt.stump


def test_multiclass_predict_uses_fitted_encoding(monkeypatch):
    # Isolate the deterministic prompt-feature contract, not LLM quality.
    class Stump:
        def __init__(self, **kwargs):
            pass

        def predict(self, X):
            return np.array([int(x) for x in X])

    monkeypatch.setattr(imodelsx.llm, 'get_llm', lambda checkpoint: SimpleNamespace(model_=None))
    monkeypatch.setattr(imodelsx.treeprompt.stump, 'PromptStump', Stump)
    model = TreePromptClassifier(checkpoint='gpt2', prompts=['prompt'],
                                 verbalizer={i: str(i) for i in range(4)},
                                 tree_kwargs={'max_leaf_nodes': 4},
                                 cache_prompt_features_dir=None)
    X = ['0', '1', '2', '3'] * 2
    y = np.array([0, 1, 2, 3] * 2)
    model.fit(X, y)
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (8, 4)
    np.testing.assert_array_equal(model.predict(X), y)
