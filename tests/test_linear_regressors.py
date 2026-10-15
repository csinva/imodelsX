import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from transformers import BertConfig, BertModel, BertTokenizer

from imodelsx import LinearNgramRegressor, LinearFinetuneRegressor, AugLinearRegressor, AugLinearClassifier


@pytest.fixture
def tiny_checkpoint(tmp_path):
    # A local, randomly initialized encoder: tests the API, not model quality.
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'good', 'bad', 'movie', 'film']
    (tmp_path / 'vocab.txt').write_text('\n'.join(vocab))
    BertTokenizer(vocab_file=str(tmp_path / 'vocab.txt'), model_max_length=16).save_pretrained(tmp_path)
    BertModel(BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=16,
                         max_position_embeddings=32)).save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize('checkpoint', ['countvectorizer', 'tfidfvectorizer'])
def test_ngram_regressor_fit_predict(checkpoint):
    X = ['good movie', 'bad film'] * 5
    y = np.tile([3., 5.], 5)
    model = LinearNgramRegressor(checkpoint=checkpoint)
    with pytest.raises(NotFittedError):
        model.predict(X)
    assert model.fit(X, y, verbose=False) is model
    np.testing.assert_allclose(model.predict(X), model.linear.predict(model.vectorizer.transform(X)))


def test_finetune_regressor_fit_predict(tiny_checkpoint):
    X = ['good movie', 'bad film'] * 5
    y = np.tile([3., 5.], 5)
    model = LinearFinetuneRegressor(checkpoint=tiny_checkpoint, device='cpu')
    with pytest.raises(NotFittedError):
        model.predict(X)
    model.fit(X, y)
    np.testing.assert_allclose(model.predict(X), model.linear.predict(model._get_embs(X)))
    assert model.predict(X[:1]).shape == (1,)


def test_auglinear_regressor_cached_predictions(tiny_checkpoint):
    X = ['good movie', 'bad film'] * 5
    y = np.tile([3., 5.], 5)
    model = AugLinearRegressor(checkpoint=tiny_checkpoint, ngrams=1, embedding_prefix='')
    assert model.fit(X, y, verbose=False) is model
    expected = np.array([
        model.coefs_dict_['good'] + model.coefs_dict_['movie'],
        model.coefs_dict_['bad'] + model.coefs_dict_['film'],
    ]) + model.linear.intercept_
    np.testing.assert_allclose(model.predict(X[:2]), expected)
    assert model.predict(X[:1]).shape == (1,)


@pytest.mark.parametrize('labels', [[3, 5], [3, 5, 7]])
def test_auglinear_classifier_scores_and_labels(tiny_checkpoint, labels):
    from scipy.special import softmax

    phrases = ['good movie', 'bad film', 'good film'][:len(labels)]
    X = phrases * 5
    y = np.tile(labels, 5)
    model = AugLinearClassifier(checkpoint=tiny_checkpoint, ngrams=1, embedding_prefix='')
    model.fit(X, y, verbose=False)
    scores = np.array([sum(model.coefs_dict_[word] for word in text.split())
                       for text in phrases]) + model.linear.intercept_
    logits = np.column_stack((np.zeros(len(phrases)), scores)) if len(labels) == 2 else scores
    expected = softmax(logits, axis=1)
    np.testing.assert_allclose(model.predict_proba(phrases), expected)
    np.testing.assert_array_equal(model.predict(phrases), np.array(labels)[expected.argmax(axis=1)])
    assert model.predict(phrases[:1]).shape == (1,)
    assert model.predict_proba(phrases[:1]).shape == (1, len(labels))
