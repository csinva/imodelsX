<p align="center">  <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="18%"> 
<img align="center" width=40% src="https://csinva.io/imodelsX/imodelsx_logo.svg?sanitize=True&kill_cache=1"> </img>	<img src="https://csinva.io/emb-gam/embgam_gif.gif" width="18%"></p>

<p align="center">Library to explain <i>a dataset</i> in natural language.
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

| Model                       | Reference                                                    | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Emb-GAM            | [🗂️](https://csinva.io/emb-gam/), [🔗](https://github.com/csinva/emb-gam), [📄](https://arxiv.org/abs/2209.11799) | Fit better linear model using an LLM to extract embeddings (*Official*) |
| iPrompt            | [🗂️](), [🔗](https://github.com/csinva/interpretable-autoprompting), [📄]() | Generates a human-interpretable prompt that explains patterns in data while still inducing strong generalization performance. (*Official*) |
| AutoPrompt            | [🗂️](), [🔗](https://github.com/ucinlp/autoprompt), [📄](https://arxiv.org/abs/2010.15980) |Find a natural-language prompt using input-gradients. |
| D3            | [🗂️](), [🔗](https://github.com/ruiqi-zhong/DescribeDistributionalDifferences), [📄](https://arxiv.org/abs/2201.12323) |Explain the difference between two distributions. |
| More models                 | ⌛                                                            | (Coming soon!) Lightweight Rule Induction, MLRules, ... |

<p align="center">
Docs <a href="https://csinva.io/imodels/">🗂️</a>, Reference code implementation 🔗, Research paper 📄
</br>
</p>


# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

**Usage example** (see <a href="https://csinva.github.io/emb-gam/">api</a> or <a href="https://github.com/csinva/emb-gam/blob/master/demo_embgam.ipynb">demo notebook</a> for more details):

```python
from imodelsx import EmbGAMClassifier, ...
``` 

# Docs
- still in progress....

# Related work
- imodels package (JOSS 2021 [github](https://github.com/csinva/imodels)) - interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).
- Adaptive wavelet distillation (NeurIPS 2021 [pdf](https://arxiv.org/abs/2107.09145), [github](https://github.com/Yu-Group/adaptive-wavelets)) - distilling a neural network into a concise wavelet model
- Transformation importance (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- Hierarchical interpretations (ICLR 2019 [pdf](https://openreview.net/pdf?id=SkEqro0ctQ), [github](https://github.com/csinva/hierarchical-dnn-interpretations)) - extends CD to CNNs / arbitrary DNNs, and aggregates explanations into a hierarchy
- Interpretation regularization (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- PDR interpretability framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning
