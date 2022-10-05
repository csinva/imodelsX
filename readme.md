<p align="center">  <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="25%"> 
<img align="center" width=49% src="https://csinva.io/imodelsX/imodelsx_logo.svg?sanitize=True&kill_cache=1"> </img>	<img src="https://csinva.io/emb-gam/embgam_gif.gif" width="25%"></p>

<p align="center">Library to explain *a dataset* in natural language.
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

- <b>Official code for using / reproducing Emb-GAM from the paper "Emb-GAM: an interpretable and efficient predictor using pre-trained language models" (<a href="https://arxiv.org/abs/2209.11799">singh & gao, 2022</a>).
</b> Emb-GAM uses a pre-trained language model to extract features from text data then combines them in order to extract out a simple, linear model.
- <b>Official code for using / reproducing iPrompt from the paper "Explaining Patterns in Data  with  Language Models via Interpretable Autoprompting" (<a href="https://arxiv.org/abs/2">Singh*, Morris*, Aneja, Rush & Gao, 2022</a>) </b> iPrompt generates a human-interpretable prompt that explains patterns in data while still inducing strong generalization performance.
- Autoprompt
- D3


# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

**Usage example** (see <a href="https://csinva.github.io/emb-gam/">api</a> or <a href="https://github.com/csinva/emb-gam/blob/master/demo_embgam.ipynb">demo notebook</a> for more details):

```python
from imodelsx import EmbGAMClassifier
import datasets
import numpy as np
``` 

# Docs
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
