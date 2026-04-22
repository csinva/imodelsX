# Explaining text data by bridging interpretable models and LLMs

<p align="center"><img src="assets/blog_anim.gif" alt="messy text goes through a magnifying glass and emerges as a simple explanation"></p>

*Explaining data* is the overaching goal of data-driven science, allowing us to discover and quantitatively test hypotheses.
The pursuit of data-driven explanations has led to the development of many interpretable models that allow a user to understand data patterns, such as decision trees, rule lists, and sparse linear models.
However, these models are often not suitable to the peculiarities of text data, which is generally sparse, high-dimensional, and full of complex interactions.
In contrast, LLMs have displayed impressive proficiency at handling text data, but they are often considered black boxes. Here, let's look at some recent work on bridging the gap between interpretable models and LLMs.

## Interpretable models

Many interpretable models have been proposed to interpret data involved in prediction problems (e.g. classification or regression). They may take slightly different forms (see some in Fig 1), but provide a complete description of the prediction process (as long as they're sufficiently accurate and small!). These models excel at tabular data, but struggle with other structured domains, such as text data.

|                           Rule set                           |                        Rule list                        |                        Rule tree                        |                       Algebraic models                       |
| :----------------------------------------------------------: | :-----------------------------------------------------: | :-----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://csinva.io/imodels/img/rule_set.jpg" class="full_image"> | <img src="https://csinva.io/imodels/img/rule_list.jpg" class="full_image"> | <img src="https://csinva.io/imodels/img/rule_tree.jpg" class="full_image"> | <img src="https://csinva.io/imodels/img/algebraic_models.jpg" class="full_image"> |

<p align="center"><b>Figure 1. </b>Different types of interpretable models. See scikit-learn friendly implementations in the <a href="https://github.com/csinva/imodels">imodels package</a>.</p>

## Adding LLMs to interpretable models

Fig 2 shows some newer model forms that seek data explanations using LLMs/ interpretable models. For text data, These models are often more accurate than their interpretable counterparts, but still provide a complete description of the prediction process.

In the most direct case, an LLM is fed data corresponding to 2 groups (binary classification) and prompted to directly produce a description of the difference between the groups ([D3](https://proceedings.mlr.press/v162/zhong22a.html)/[D5](https://arxiv.org/abs/2302.14233)).
Alternatively, given a dataset and a pre-trained LLM, [iPrompt](https://arxiv.org/abs/2210.01848) searches for a natural-language prompt that works well to predict on the dataset, which serves as a description of the data. This is more general than D3, as it is not restricted to binary groups, but is also more computationally intensive, as finding a good prompt often requires iterative LLM calls.
Either of these approaches can also be applied recursively ([TreePrompt](https://arxiv.org/abs/2310.14034)), resulting in a hierarchical natural-language description of the data.
Alternatively, many LLM answers to different questions can be concatenated into an embedding ([QA-Emb](https://arxiv.org/abs/2405.16714)), potentially incorporating bayesian iteration ([BC-LLM](https://arxiv.org/abs/2410.15555)), which can then be used to train a fully interpretable model, e.g. a linear model.

<img src="assets/interpretable_models.svg" class="full_image">
<p align="center" style="margin-top:-20px"><b>Figure 2. </b>Different types of interpretable models, with text-specific approaches in bold. See scikit-learn friendly implementations below.</p>

In parallel to these methods, [Aug-imodels](https://arxiv.org/abs/2209.11799) use LLMs to improve fully interpretable models directly.
For example, Aug-Linear uses an LLM to augment a linear model, resulting in a more accurate model that is still completely interpretable. This is done by using an LLM *only during training time* to generate a dictionary of coefficients that is then extremely efficient and interpretable at inference time, while still maintaining reasonably high prediction accuracy (see Fig 3).

<img src="assets/auglinear.png" class="full_image">
<p align="center" style="margin-top:-20px"><b>Figure 3. </b>Aug-Linear uses an LLM to augment a linear model, resulting in a more accurate model that is still completely interpretable. The LLM is only used during training to generate a dictionary of coefficients, which is then used for efficient and interpretable inference.</p>

The key to many of these explanation methods is finding ways to evaluate hypotheses without groundtruth, e.g. testing on [follow-up experiments](https://arxiv.org/abs/2410.00812), [synthetic data](https://arxiv.org/abs/2305.09863), [prior findings](https://www.biorxiv.org/content/10.1101/2025.08.12.669958v1), [held-out data](https://aclanthology.org/2023.blackboxnlp-1.3/), [counterfactuals](https://aclanthology.org/2025.coling-main.505.pdf), or [new LLMs](https://arxiv.org/abs/2601.11517).   This line of research is still in its infancy -- there's a lot to be done in combining LLMs and interpretable models!

# Code reference below
