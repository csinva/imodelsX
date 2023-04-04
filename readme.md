<p align="center">  <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="18%"> 
<img align="center" width=40% src="https://csinva.io/imodelsX/imodelsx_logo.svg?sanitize=True&kill_cache=1"> </img>	<img src="https://csinva.io/emb-gam/embgam_gif.gif" width="18%"></p>

<p align="center">Library to explain <i>a dataset</i> in natural language. 
</p>
<p align="center">
  <a href="https://github.com/csinva/imodelsX/tree/master/demo_notebooks">📖 demo notebooks</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6+-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">  
</p>  

| Model                       | Reference                                                    | Output  | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| iPrompt            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/iprompt.ipynb), [🗂️](http://csinva.io/imodelsX/iprompt/api.html#imodelsx.iprompt.api.explain_dataset_iprompt), [🔗](https://github.com/csinva/interpretable-autoprompting), [📄](https://arxiv.org/abs/2210.01848) | Explanation | Generates a human-interpretable prompt that<br/>explains patterns in data (*Official*) |
| D3            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/d3.py), [🗂️](http://csinva.io/imodelsX/d3/d3.html#imodelsx.d3.d3.explain_datasets_d3), [🔗](https://github.com/ruiqi-zhong/DescribeDistributionalDifferences), [📄](https://arxiv.org/abs/2201.12323) | Explanation | Explain the difference between two distributions |
| AutoPrompt            | ⠀⠀⠀[🗂️](), [🔗](https://github.com/ucinlp/autoprompt), [📄](https://arxiv.org/abs/2010.15980) | Explanation | Find a natural-language prompt using input-gradients (⌛ In progress)|
| Emb-GAM            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/embgam.ipynb), [🗂️](http://csinva.io/imodelsX/embgam/embgam.html#imodelsx.embgam.embgam.EmbGAMClassifier), [🔗](https://github.com/csinva/emb-gam), [📄](https://arxiv.org/abs/2209.11799) | Linear model | Fit better linear model using an LLM to extract embeddings (*Official*) |
| Linear Finetune  | ⠀⠀⠀[🗂️](http://csinva.io/imodelsX/linear_finetune.html) | Black-box model | Scikit-learn interface to finetune a single linear layer<br/>on top of LLM embeddings for classification/regression |
| (Coming soon!)                 | ⌛                                        |                    |  We plan to support other interpretable models like [RLPrompt](https://arxiv.org/abs/2205.12548), <br/> [concept bottleneck models](https://arxiv.org/abs/2007.04612), [NAMs](https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html), and [NBDT](https://arxiv.org/abs/2004.00221)  

<p align="center">
Demo notebooks <a href="https://github.com/csinva/imodelsX/tree/master/demo_notebooks">📖</a>, Doc <a href="https://csinva.io/imodelsX/">🗂️</a>, Reference code implementation 🔗, Research paper 📄
</br>
</p>


# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

**Demos**: see the [demo notebooks](https://github.com/csinva/imodelsX/tree/master/demo_notebooks)

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
import imodelsx
hypotheses, hypothesis_scores = imodelsx.explain_datasets_d3(
    pos=positive_samples, # List[str] of positive examples
    neg=negative_samples, # another List[str]
    num_steps=100,
    num_folds=2,
    batch_size=64,
)
```

### Emb-GAM

```python
from imodelsx import EmbGAMClassifier
import datasets
import numpy as np

# set up data
dset = datasets.load_dataset('rotten_tomatoes')['train']
dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
dset_val = dset_val.select(np.random.choice(len(dset_val), size=300, replace=False))

# fit model
m = EmbGAMClassifier(
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

# Related work
- imodels package (JOSS 2021 [github](https://github.com/csinva/imodels)) - interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).
- Adaptive wavelet distillation (NeurIPS 2021 [pdf](https://arxiv.org/abs/2107.09145), [github](https://github.com/Yu-Group/adaptive-wavelets)) - distilling a neural network into a concise wavelet model
- Transformation importance (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- Hierarchical interpretations (ICLR 2019 [pdf](https://openreview.net/pdf?id=SkEqro0ctQ), [github](https://github.com/csinva/hierarchical-dnn-interpretations)) - extends CD to CNNs / arbitrary DNNs, and aggregates explanations into a hierarchy
- Interpretation regularization (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- PDR interpretability framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning
