'''End-to-end tests that call real models. Skipped unless the checkpoints are given through env vars:

    IMODELSX_TEST_HF_CHECKPOINT   local huggingface model (e.g. Qwen/Qwen2.5-0.5B-Instruct or a local path)
    IMODELSX_TEST_API_CHECKPOINT  any API / CLI checkpoint (e.g. claude-code:haiku, gpt-4o-mini)

Example:
    IMODELSX_TEST_HF_CHECKPOINT=~/hf_models/Qwen2.5-0.5B-Instruct \
    IMODELSX_TEST_API_CHECKPOINT=claude-code:haiku pytest tests/test_live.py -s
Generation-only methods run with both checkpoints; methods that need logits / embeddings run with the hf one.
'''
import os
from os.path import expanduser

import numpy as np
import pytest

import imodelsx
import imodelsx.llm

HF = os.environ.get('IMODELSX_TEST_HF_CHECKPOINT')
API = os.environ.get('IMODELSX_TEST_API_CHECKPOINT')
HF = expanduser(HF) if HF and HF.startswith('~') else HF
needs_hf = pytest.mark.skipif(not HF, reason='set IMODELSX_TEST_HF_CHECKPOINT')
both = pytest.mark.parametrize('kind', [
    pytest.param('hf', marks=needs_hf),
    pytest.param('api', marks=pytest.mark.skipif(not API, reason='set IMODELSX_TEST_API_CHECKPOINT')),
])

_LLMS = {}


def get_checkpoint(kind):
    return HF if kind == 'hf' else API


def get_llm(kind, tmp_path_factory):
    '''Load each model once per session (with a fresh cache dir)'''
    if kind not in _LLMS:
        _LLMS[kind] = imodelsx.llm.get_llm(
            get_checkpoint(kind), CACHE_DIR=str(tmp_path_factory.mktemp(f'cache_{kind}')))
    return _LLMS[kind]


@both
def test_llm_generate(kind, tmp_path_factory):
    llm = get_llm(kind, tmp_path_factory)
    if kind == 'hf':
        out = llm('The capital of Spain is the city of', max_new_tokens=5)
    else:
        out = llm('What is the capital of Spain? Answer with one word.', verbose=False)
    print(f'\n[{kind}] capital of Spain ->', repr(out))
    assert 'madrid' in out.lower()

    # lists of prompts return lists, and repeated calls hit the cache
    prompts = ['Name a primary color.', 'Name a fruit.']
    outs = llm(prompts, max_new_tokens=10) if kind == 'hf' else llm(prompts, verbose=False)
    print(f'[{kind}] batch ->', outs)
    assert isinstance(outs, list) and len(outs) == 2 and all(isinstance(o, str) for o in outs)
    outs_cached = llm(prompts, max_new_tokens=10) if kind == 'hf' else llm(prompts, verbose=False)
    assert outs_cached == outs


@needs_hf
def test_llm_next_token_probs(tmp_path_factory):
    llm = get_llm('hf', tmp_path_factory)
    prompts = ['Roses are red, violets are', 'The opposite of hot is']
    probs = llm(prompts, return_next_token_prob_scores=True,
                target_token_strs=[' blue', ' cold'], use_cache=False)
    print('\nnext-token probs ->', probs)
    assert probs[0][' blue'] > probs[0][' cold'] and probs[1][' cold'] > probs[1][' blue']
    top = llm(prompts, return_top_target_token_str=True,
              target_token_strs=[' blue', ' cold'], use_cache=False)
    assert top == [' blue', ' cold']


@both
def test_qaemb(kind, tmp_path):
    from imodelsx.qaemb.qaemb import get_sample_questions_and_examples
    questions, examples = get_sample_questions_and_examples()
    embedder = imodelsx.QAEmb(questions=questions, checkpoint=get_checkpoint(kind),
                              use_cache=False, CACHE_DIR=str(tmp_path), batch_size=4)
    embs = embedder(examples, verbose=False)
    # example i answers yes to question i only
    acc = np.mean(embs == np.eye(len(examples)))
    print(f'\n[{kind}] qaemb embeddings\n{embs}\nagreement with expected answers: {acc:.2f}')
    assert embs.shape == (len(examples), len(questions))
    if kind == 'api':
        assert acc >= 0.8


@both
def test_sasc(kind, tmp_path):
    # module responds to string length; the longest strings are animals
    text_str_list = ['red', 'blue', 'x', '1', '2', 'hippopotamus', 'elephant', 'rhinoceros']
    explanation_dict = imodelsx.explain_module_sasc(
        text_str_list,
        lambda str_list: np.array([len(s) for s in str_list]),
        ngrams=1,
        num_summaries=2,
        num_top_ngrams=3,
        num_top_ngrams_to_consider=3,
        num_synthetic_strs=2,
        llm_checkpoint=get_checkpoint(kind),
        llm_cache_dir=str(tmp_path),
        use_cache=False,
    )
    print(f'\n[{kind}] sasc explanations ->', explanation_dict['explanation_strs'],
          '\nexplanation scores ->', explanation_dict['explanation_scores'])
    assert len(explanation_dict['explanation_strs']) >= 1
    if kind == 'api':
        assert any('animal' in s.lower() or 'mammal' in s.lower()
                   for s in explanation_dict['explanation_strs'])


@both
def test_augtree_expand_keyword(kind):
    import imodelsx.augtree.llm
    keywords = imodelsx.augtree.llm.expand_keyword(
        'awful', ' in the context of movie reviews', checkpoint=get_checkpoint(kind))
    print(f'\n[{kind}] expansions of "awful" ->', keywords[:15])
    assert len(keywords) >= 5


def _tiny_rotten_tomatoes(n=40, seed=0):
    import datasets
    rng = np.random.default_rng(seed)
    dset = datasets.load_dataset('rotten_tomatoes')
    train = dset['train'].select(rng.choice(len(dset['train']), size=n, replace=False))
    val = dset['validation'].select(rng.choice(len(dset['validation']), size=n, replace=False))
    return train['text'], np.array(train['label']), val['text'], np.array(val['label'])


@needs_hf
def test_treeprompt():
    X, y, X_val, y_val = _tiny_rotten_tomatoes()
    m = imodelsx.TreePromptClassifier(
        checkpoint=HF,
        prompts=[' The sentiment of the movie review was', ' Overall, the movie was'],
        verbalizer={0: ' Negative.', 1: ' Positive.'},
        cache_prompt_features_dir=None,
        device='cpu', batch_size=8,
    )
    m.fit(X, y)
    acc = np.mean(m.predict(X_val) == y_val)
    print(f'\ntreeprompt val acc {acc:.2f}, per-prompt accs {m.prompt_accs_}')
    assert acc > 0.5


@needs_hf
def test_linear_finetune():
    X, y, X_val, y_val = _tiny_rotten_tomatoes()
    m = imodelsx.LinearFinetuneClassifier(checkpoint=HF, device='cpu')
    m.fit(X, y)
    acc = np.mean(m.predict(X_val) == y_val)
    print(f'\nlinear finetune val acc {acc:.2f}')
    assert acc > 0.5


@needs_hf
def test_auglinear():
    X, y, X_val, y_val = _tiny_rotten_tomatoes(n=20)
    m = imodelsx.AugLinearClassifier(checkpoint=HF, ngrams=1, min_frequency=1)
    m.fit(X, y, batch_size=8)
    acc = np.mean(m.predict(X_val) == y_val)
    top = sorted(m.coefs_dict_.items(), key=lambda kv: kv[1])
    print(f'\nauglinear val acc {acc:.2f}; most negative {top[:3]}; most positive {top[-3:]}')
    assert len(m.coefs_dict_) > 0


@needs_hf
def test_embeddings():
    from imodelsx.embeddings import get_embs
    embs = get_embs(['a great movie', 'a terrible movie'], checkpoint=HF, batch_size=2)
    print('\nembeddings shape', embs.shape)
    assert embs.shape[0] == 2 and np.isfinite(embs).all()


@needs_hf
def test_iprompt(tmp_path):
    input_strings, output_strings = imodelsx.get_add_two_numbers_dataset(num_examples=20)
    prompts, metadata = imodelsx.explain_dataset_iprompt(
        input_strings=input_strings,
        output_strings=output_strings,
        checkpoint=HF,
        num_learned_tokens=3,
        n_shots=1,
        n_epochs=1,
        max_n_steps=4,
        batch_size=4,
        llm_float16=False,
        save_dir=str(tmp_path),
        verbose=0,
    )
    print('\niprompt top prompts ->', prompts[:3])
    assert len(prompts) > 0
