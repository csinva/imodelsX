<p align="center">  <img src="https://microsoft.github.io/aug-models/embgam_gif.gif" width="18%"> 
<img align="center" width=40% src="https://csinva.io/imodelsX/imodelsx_logo.svg?sanitize=True&kill_cache=1"> </img>	<img src="https://microsoft.github.io/aug-models/embgam_gif.gif" width="18%"></p>

<p align="center">Scikit-learn friendly library to explain, predict, and steer text models/data.<br/>Also a bunch of utilities for getting started with text data.
</p>
<p align="center">
  <a href="https://github.com/csinva/imodelsX/tree/master/demo_notebooks">📖 demo notebooks</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">  
</p>  


**Explainable modeling/steering**

| Model                       | Reference                                                    | Output  | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| Tree-Prompt            | [🗂️](http://csinva.io/imodelsX/treeprompt/treeprompt.html), [🔗](https://github.com/csinva/tree-prompt/tree/main), [📄](https://arxiv.org/abs/2310.14034), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/tree_prompt.ipynb),  | Explanation<br/>+ Steering | Generates a tree of prompts to<br/>steer an LLM (*Official*) |
| iPrompt            | [🗂️](http://csinva.io/imodelsX/iprompt/api.html#imodelsx.iprompt.api.explain_dataset_iprompt), [🔗](https://github.com/csinva/interpretable-autoprompting), [📄](https://arxiv.org/abs/2210.01848), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/iprompt.ipynb) | Explanation<br/>+ Steering | Generates a prompt that<br/>explains patterns in data (*Official*) |
| AutoPrompt            | ㅤㅤ[🗂️](), [🔗](https://github.com/ucinlp/autoprompt), [📄](https://arxiv.org/abs/2010.15980) | Explanation<br/>+ Steering | Find a natural-language prompt<br/>using input-gradients|
| D3            | [🗂️](http://csinva.io/imodelsX/d3/d3.html#imodelsx.d3.d3.explain_dataset_d3), [🔗](https://github.com/ruiqi-zhong/DescribeDistributionalDifferences), [📄](https://arxiv.org/abs/2201.12323), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/d3.ipynb) | Explanation | Explain the difference between two distributions |
| SASC            |   ㅤㅤ[🗂️](https://csinva.io/imodelsX/sasc/api.html), [🔗](https://github.com/microsoft/automated-explanations), [📄](https://arxiv.org/abs/2305.09863) | Explanation | Explain a black-box text module<br/>using an LLM (*Official*) |
| Aug-Linear            | [🗂️](https://csinva.io/imodelsX/auglinear/auglinear.html), [🔗](https://github.com/microsoft/aug-models), [📄](https://www.nature.com/articles/s41467-023-43713-1), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/aug_imodels.ipynb) | Linear model | Fit better linear model using an LLM<br/>to extract embeddings (*Official*) |
| Aug-Tree            | [🗂️](https://csinva.io/imodelsX/augtree/augtree.html), [🔗](https://github.com/microsoft/aug-models), [📄](https://www.nature.com/articles/s41467-023-43713-1), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/aug_imodels.ipynb) | Decision tree | Fit better decision tree using an LLM<br/>to expand features (*Official*) |
| QAEmb            | [🗂️](https://csinva.io/imodelsX/qaemb/qaemb.html), [🔗](https://github.com/csinva/interpretable-embeddings), [📄](https://arxiv.org/abs/2405.16714), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/qaemb.ipynb) | Explainable<br/>embedding | Generate interpretable embeddings<br/>by asking LLMs questions (*Official*) |
| KAN            | [🗂️](https://csinva.io/imodelsX/kan/kan_sklearn.html), [🔗](https://github.com/Blealtan/efficient-kan/tree/master), [📄](https://arxiv.org/abs/2404.19756), [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/kan.ipynb) | Small<br/>network | Fit 2-layer Kolmogorov-Arnold network |

<p align="center">
<a href="https://github.com/csinva/imodelsX/tree/master/demo_notebooks">📖</a>Demo notebooks &emsp; <a href="https://csinva.io/imodelsX/">🗂️</a> Doc &emsp; 🔗 Reference code &emsp; 📄 Research paper
</br>
⌛ We plan to support other interpretable algorithms like <a href="https://arxiv.org/abs/2205.12548">RLPrompt</a>, <a href="https://arxiv.org/abs/2007.04612">CBMs</a>, and <a href="https://arxiv.org/abs/2004.00221">NBDT</a>. If you want to contribute an algorithm, feel free to open a PR 😄
</p>

**General utilities**

| Model                       | Reference                                                    | 
| :-------------------------- | ------------------------------------------------------------ | 
|  [🗂️](https://csinva.io/imodelsX/llm.html)  LLM wrapper| Easily call different LLMs |
|  [🗂️](https://csinva.io/imodelsX/data.html)  Dataset wrapper| Download minimially processed huggingface datasets |
| [🗂️](https://csinva.io/imodelsX/linear_ngram.html) Bag of Ngrams    | Learn a linear model of ngrams |
| [🗂️](https://csinva.io/imodelsX/linear_finetune.html) Linear Finetune  | Finetune a single linear layer on top of LLM embeddings |


# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

<details>
<summary><b>Set up your LLM</b>: use local huggingface models or API models from OpenAI, Anthropic, Claude Code, or Codex</summary>

Every method that takes a `checkpoint` / `llm_checkpoint` calls LLMs through `imodelsx.llm.get_llm`.
Pick a provider by prefixing the checkpoint with `provider:` (or let it be inferred from the name):

| Provider | Example checkpoint | Credentials | Install |
|---|---|---|---|
| `hf` (default) | `gpt2`, `Qwen/Qwen2.5-0.5B-Instruct`, a local path | `HF_TOKEN` (only for gated models) | included |
| `hf-api` | `hf-api:meta-llama/Llama-3.1-8B-Instruct` | `HF_TOKEN` | included |
| `openai` | `gpt-4o-mini`, `openai:gpt-5` | `OPENAI_API_KEY` (`OPENAI_BASE_URL` for compatible servers) | `pip install "imodelsx[openai]"` |
| `azure` | `azure:gpt-4o` | `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY`, or Azure CLI login | `pip install "imodelsx[openai]"` |
| `anthropic` | `claude-haiku-5-5`, `anthropic:claude-opus-5-5` | `ANTHROPIC_API_KEY` | `pip install "imodelsx[anthropic]"` |
| `claude-code` | `claude-code:haiku`, `claude-code:sonnet` | your Claude Code login (no key needed) | [Claude Code](https://code.claude.com) CLI |
| `codex` | `codex:gpt-5-codex`, `codex:default` | your Codex login (`codex login`) or `OPENAI_API_KEY` | `npm i -g @openai/codex` |

Keys are read from the `api_key` argument, then the environment variable, then a file at `~/.<ENV_VAR>` (e.g. `~/.OPENAI_API_KEY`).
Without a prefix, `gpt-*`/`o*` names use `openai` and `claude-*` names use `anthropic` (falling back to `azure` / `claude-code` when no key is found); anything else loads a local huggingface model.

```python
import imodelsx.llm
llm = imodelsx.llm.get_llm('claude-code:haiku')  # or 'gpt-4o-mini', 'Qwen/Qwen2.5-0.5B-Instruct', ...
llm('What is the capital of Spain?')  # outputs are cached in ~/.CACHE_LLM
llm(['prompt 1', 'prompt 2'])  # API models run lists of prompts in parallel threads
llm = imodelsx.llm.get_llm('gpt-4o-mini', api_key='sk-...')  # pass a key directly
```

Methods that need next-token probabilities or embeddings (e.g. Tree-Prompt, Aug-Linear, linear finetuning) require a local `hf` model; methods that only generate text (e.g. SASC, QA-Emb, Aug-Tree keyword expansion) work with any provider.
</details>


**Demos**: see the [demo notebooks](https://github.com/csinva/imodelsX/tree/master/demo_notebooks)


# Natural-language explanations

### Tree-prompt
```python
from imodelsx import TreePromptClassifier
import datasets
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# set up data
rng = np.random.default_rng(seed=42)
dset_train = datasets.load_dataset('rotten_tomatoes')['train']
dset_train = dset_train.select(rng.choice(
    len(dset_train), size=100, replace=False))
dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
dset_val = dset_val.select(rng.choice(
    len(dset_val), size=100, replace=False))

# set up arguments
prompts = [
    "This movie is",
    " Positive or Negative? The movie was",
    " The sentiment of the movie was",
    " The plot of the movie was really",
    " The acting in the movie was",
]
verbalizer = {0: " Negative.", 1: " Positive."}
checkpoint = "gpt2"

# fit model
m = TreePromptClassifier(
    checkpoint=checkpoint,
    prompts=prompts,
    verbalizer=verbalizer,
    cache_prompt_features_dir=None,  # 'cache_prompt_features_dir/gp2',
)
m.fit(dset_train["text"], dset_train["label"])


# compute accuracy
preds = m.predict(dset_val['text'])
print('\nTree-Prompt acc (val) ->',
      np.mean(preds == dset_val['label']))  # -> 0.7

# compare to accuracy for individual prompts
for i, prompt in enumerate(prompts):
    print(i, prompt, '->', m.prompt_accs_[i])  # -> 0.65, 0.5, 0.5, 0.56, 0.51

# visualize decision tree
plot_tree(
    m.clf_,
    fontsize=10,
    feature_names=m.feature_names_,
    class_names=list(verbalizer.values()),
    filled=True,
)
plt.show()
```

### iPrompt

```python
from imodelsx import explain_dataset_iprompt, get_add_two_numbers_dataset

# get a simple dataset of adding two numbers
input_strings, output_strings = get_add_two_numbers_dataset(num_examples=100)
for i in range(5):
    print(repr(input_strings[i]), repr(output_strings[i]))

# explain the relationship between the inputs and outputs
# with a natural-language prompt string
prompts, metadata = explain_dataset_iprompt(
    input_strings=input_strings,
    output_strings=output_strings,
    checkpoint='EleutherAI/gpt-j-6B', # which language model to use
    num_learned_tokens=3, # how long of a prompt to learn
    n_shots=3, # shots per example
    n_epochs=15, # how many epochs to search
    verbose=0, # how much to print
    llm_float16=True, # whether to load the model in float_16
)
--------
prompts is a list of found natural-language prompt strings
```

### D3 (DescribeDistributionalDifferences)

```python
from imodelsx import explain_dataset_d3
hypotheses, hypothesis_scores = explain_dataset_d3(
    pos=positive_samples, # List[str] of positive examples
    neg=negative_samples, # another List[str]
    num_steps=100,
    num_folds=2,
    batch_size=64,
)
```

### SASC
Here, we explain a *module* rather than a dataset

```python
from imodelsx import explain_module_sasc
# a toy module that responds to the length of a string
mod = lambda str_list: np.array([len(s) for s in str_list])

# a toy dataset where the longest strings are animals
text_str_list = ["red", "blue", "x", "1", "2", "hippopotamus", "elephant", "rhinoceros"]
explanation_dict = explain_module_sasc(
    text_str_list,
    mod,
    ngrams=1,
)
```

# Aug-imodels
Use these just a like a scikit-learn model. During training, they fit better features via LLMs, but at test-time they are extremely fast and completely transparent.

```python
from imodelsx import AugLinearClassifier, AugTreeClassifier, AugLinearRegressor, AugTreeRegressor
import datasets
import numpy as np

# set up data
dset = datasets.load_dataset('rotten_tomatoes')['train']
dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
dset_val = dset_val.select(np.random.choice(len(dset_val), size=300, replace=False))

# fit model
m = AugLinearClassifier(
    checkpoint='textattack/distilbert-base-uncased-rotten-tomatoes',
    ngrams=2, # use bigrams
)
m.fit(dset['text'], dset['label'])

# predict
preds = m.predict(dset_val['text'])
print('acc_val', np.mean(preds == dset_val['label']))

# interpret
print('Total ngram coefficients: ', len(m.coefs_dict_))
print('Most positive ngrams')
for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1], reverse=True)[:8]:
    print('\t', k, round(v, 2))
print('Most negative ngrams')
for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1])[:8]:
    print('\t', k, round(v, 2))
```

# KAN
```python
import imodelsx
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_classification(n_samples=5000, n_features=5, n_informative=3)
model = imodelsx.KANClassifier(hidden_layer_size=64, device='cpu',
                               regularize_activation=1.0, regularize_entropy=1.0)
model.fit(X, y)
y_pred = model.predict(X)
print('Test acc', accuracy_score(y, y_pred))

# now try regression
X, y = make_regression(n_samples=5000, n_features=5, n_informative=3)
model = imodelsx.kan.KANRegressor(hidden_layer_size=64, device='cpu',
                                  regularize_activation=1.0, regularize_entropy=1.0)
model.fit(X, y)
y_pred = model.predict(X)
print('Test correlation', np.corrcoef(y, y_pred.flatten())[0, 1])
```


# General utilities

### Easy baselines
Easy-to-fit baselines that follow the sklearn API.

```python
from imodelsx import LinearFinetuneClassifier, LinearNgramClassifier
# fit a simple one-layer finetune on top of LLM embeddings
m = LinearFinetuneClassifier(
    checkpoint='distilbert-base-uncased',
)
m.fit(dset['text'], dset['label'])
preds = m.predict(dset_val['text'])
acc = (preds == dset_val['label']).mean()
print('validation acc', acc)
```

### LLM wrapper
Easy API for calling different language models with caching (much more lightweight than [langchain](https://github.com/langchain-ai/langchain)).

```python
import imodelsx.llm
# supports huggingface, openai, anthropic, claude code, and codex checkpoints (see LLM providers in the quickstart)
llm = imodelsx.llm.get_llm(
    checkpoint="gpt2-xl",  # gpt-4o-mini, claude-haiku-5-5, claude-code:haiku, codex:gpt-5-codex, ...
    CACHE_DIR=".cache",
)
out = llm("May the Force be")
llm("May the Force be") # when computing the same string again, uses the cache
```

### Data wrapper
API for loading huggingface datasets with basic preprocessing.
```python
import imodelsx.data
dset, dataset_key_text = imodelsx.data.load_huggingface_dataset('ag_news')
# Ensures that dset has a split named 'train' and 'validation',
# and that the input data is contained for each split in a column given by {dataset_key_text}
```

# Related work
- imodels package (JOSS 2021 [github](https://github.com/csinva/imodels)) - interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).
- Rethinking Interpretability in the Era of Large Language Models (arXiv 2024 [pdf](https://arxiv.org/abs/2402.01761)) - overview of using LLMs to interpret datasets and yield natural-language explanations
- Experiments in using clinical rule development: [github](https://github.com/csinva/clinical-rule-development)
- Experiments in automatically generating brain explanations: [github](https://github.com/microsoft/automated-brain-explanations)
- Interpretation regularization (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
