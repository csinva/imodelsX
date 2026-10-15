import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from imodelsx import KANClassifier, KANRegressor, KANGAMClassifier, KANGAMRegressor
import imodelsx.kan.kan_sklearn as kan_sklearn


@pytest.mark.parametrize("estimator", [KANClassifier, KANRegressor, KANGAMClassifier, KANGAMRegressor])
@pytest.mark.parametrize("hidden_layers", [None, [2, 2]])
def test_split_parameters_and_predict(estimator, hidden_layers, monkeypatch):
    model = estimator(hidden_layer_size=2, hidden_layer_sizes=hidden_layers,
                      test_size=0.25, random_state=7, shuffle=False, device="cpu")
    model = clone(model)
    calls = []

    def record_split(*arrays, **kwargs):
        calls.append(kwargs)
        return train_test_split(*arrays, **kwargs)

    monkeypatch.setattr(kan_sklearn, "train_test_split", record_split)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(16, 2)).astype(np.float32)
    classifier = estimator in [KANClassifier, KANGAMClassifier]
    y = np.tile([0, 1], 8) if classifier else X[:, 0]
    # Keep the real optimizer/model path, but bound the CPU fixture to one epoch.
    monkeypatch.setattr(kan_sklearn, "tqdm", lambda epochs: [0])
    assert model.fit(X, y, batch_size=12) is model
    assert calls == [dict(test_size=0.25, random_state=7, shuffle=False)]
    predictions = model.predict(X[:2])
    assert predictions.shape == ((2,) if classifier else (2, 1))
    assert np.isfinite(predictions).all()
    if classifier:
        probabilities = model.predict_proba(X[:2])
        assert probabilities.shape == (2, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


def test_default_parameters_clone():
    model = clone(KANClassifier())
    assert model.get_params()["test_size"] == 0.2
    assert model.get_params()["random_state"] == 42
    assert model.get_params()["shuffle"] is True
    assert model.get_params()["hidden_layer_sizes"] is None
