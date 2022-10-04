<h1 align="center">   <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="30%"> imodelsX: interpretability for teXt <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="30%"></h1>
<p align="center"> Interpretable linear model that leverages a pre-trained language model to better learn interactions. One-line fit function.
</p>
<p align="center">
  <a href="https://csinva.github.io/emb-gam/">📚 sklearn-friendly api</a> •
  <a href="https://github.com/csinva/emb-gam/blob/master/demo_embgam.ipynb">📖 demo notebook</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">  
</p>  


<img src="https://csinva.io/emb-gam/intro_emb_gam.svg?sanitize=True">

<b>Official code for using / reproducing Emb-GAM from the paper "Emb-GAM: an interpretable and efficient predictor using pre-trained language models" (<a href="https://arxiv.org/abs/2209.11799">singh & gao, 2022</a>).
</b> Emb-GAM uses a pre-trained language model to extract features from text data then combines them in order to extract out a simple, linear model.

# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

**Usage example** (see <a href="https://csinva.github.io/emb-gam/">api</a> or <a href="https://github.com/csinva/emb-gam/blob/master/demo_embgam.ipynb">demo notebook</a> for more details):

```python
from embgam import EmbGAMClassifier
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

# Docs
<blockquote>
<b>Abstract</b>: Deep learning models have achieved impressive prediction performance but often sacrifice interpretability, a critical consideration in high-stakes domains such as healthcare or policymaking.
In contrast, generalized additive models (GAMs) can maintain interpretability but often suffer from poor prediction performance due to their inability to effectively capture feature interactions.
In this work, we aim to bridge this gap by using pre-trained large-language models to extract embeddings for each input before learning a linear model in the embedding space.
The final model (which we call Emb-GAM) is a transparent, linear function of its input features and feature interactions.
Leveraging the language model allows \methods to learn far fewer linear coefficients, model larger interactions, and generalize well to novel inputs (e.g. unseen ngrams in text).
Across a variety of natural-language-processing datasets, Emb-GAM achieves strong prediction performance without sacrificing interpretability.</blockquote>

- the main api requires simply importing `embgam.EmbGAMClassifier` or `embgam.EmbGAMRegressor`
- the `experiments` and `scripts` folder contains hyperparameters for running sweeps contained in the paper
- the `notebooks` folder contains notebooks for analyzing the outputs + making figures
- stored outputs after running all experiments are available in [this gdrive folder](https://drive.google.com/file/d/1C5ooDIlFdPxROufWWjlPr4Wmx8hDYBnh/view?usp=sharing)

# Related work
- imodels package (JOSS 2021 [github](https://github.com/csinva/imodels)) - interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).
- Adaptive wavelet distillation (NeurIPS 2021 [pdf](https://arxiv.org/abs/2107.09145), [github](https://github.com/Yu-Group/adaptive-wavelets)) - distilling a neural network into a concise wavelet model
- Transformation importance (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- Hierarchical interpretations (ICLR 2019 [pdf](https://openreview.net/pdf?id=SkEqro0ctQ), [github](https://github.com/csinva/hierarchical-dnn-interpretations)) - extends CD to CNNs / arbitrary DNNs, and aggregates explanations into a hierarchy
- Interpretation regularization (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- PDR interpretability framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning


If this package is useful for you, please cite the following!

```r
@article{singh2022embgam,
  title = {Emb-GAM: an Interpretable and Efficient Predictor using Pre-trained Language Models},
  author = {Singh, Chandan and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2209.11799},
  doi = {10.48550/arxiv.2209.11799},
  url = {https://arxiv.org/abs/2209.11799},
  year = {2022},
}

```
