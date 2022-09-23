<h1 align="center">   <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="15%"> Emb-GAM <img src="https://csinva.io/emb-gam/embgam_gif.gif" width="15%"></h1>
<p align="center"> Interpretable linear model that leverages a pre-trained language model to better learn interactions.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
</p>  


<b>Official code for using / reproducing Emb-GAM from the paper "Emb-GAM: an interpretable and efficient predictor using pre-trained language models" (<a href="https://arxiv.org/abs/2107.09145">singh & gao, 2022</a>).
</b>

<img src="https://csinva.io/emb-gam/intro_emb_gam.svg?sanitize=True">

<blockquote>
<b>Abstract</b>: Deep learning models have achieved impressive prediction performance but often sacrifice interpretability, a critical consideration in high-stakes domains such as healthcare or policymaking.
In contrast, generalized additive models (GAMs) can maintain interpretability but often suffer from poor prediction performance due to their inability to effectively capture feature interactions.
In this work, we aim to bridge this gap by using pre-trained large-language models to extract embeddings for each input before learning a linear model in the embedding space.
The final model (which we call Emb-GAM) is a transparent, linear function of its input features and feature interactions.
Leveraging the language model allows \methods to learn far fewer linear coefficients, model larger interactions, and generalize well to novel inputs (e.g. unseen ngrams in text).
Across a variety of natural-language-processing datasets, Emb-GAM achieves strong prediction performance without sacrificing interpretability.</blockquote>


# Docs
- the main python scripts are in the root directory
- the `notebooks` folder contains notebooks for analyzing the outputs + making figures
- the `scripts` folder contains hyperparameters for running sweeps
- stored outputs after running all experiments are available in [this gdrive folder](https://drive.google.com/file/d/1C5ooDIlFdPxROufWWjlPr4Wmx8hDYBnh/view?usp=sharing)

# Related work

- Adaptive wavelet distillation (NeurIPS 2021 [pdf](https://arxiv.org/abs/2107.09145), [github](https://github.com/Yu-Group/adaptive-wavelets)) - distilling a neural network into a concise wavelet model
- Transformation importance (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- Hierarchical interpretations (ICLR 2019 [pdf](https://openreview.net/pdf?id=SkEqro0ctQ), [github](https://github.com/csinva/hierarchical-dnn-interpretations)) - extends CD to CNNs / arbitrary DNNs, and aggregates explanations into a hierarchy
- Interpretation regularization (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- Disentangled attribution curves (arXiv 2019 [pdf](https://arxiv.org/abs/1905.07631), [github](https://github.com/csinva/disentangled-attribution-curves)) - finds disentangled interpretations for random forests
- PDR interpretability framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning


If this package is useful for you, please cite the following!

```r
```
