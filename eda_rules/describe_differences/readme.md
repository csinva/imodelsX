Written by Ruiqi Zhong, on 05/14/2022

### Overview

This folder contains an "end2end" implementation of the our paper: "[Describing Differences between Text Distributions with Natural Language](https://arxiv.org/abs/2201.12323)." We call it **D3** for short (**D**escribe **D**istributional **D**ifferences). Notice that this implementation is different from that in the paper, so the performance is not directly comparable (especially because we replaced the GPT-3 proposer with a T5 model, which can be open-sourced). However, we found the system to perform reasonably well.

### Caution

The system is still under development, and we would imagine researchers rather than practioners to benefit the most from this version of the system. 
In particular, please do **NOT** use it for any high-stake scenarios. We are continue working on this system to improve it, though, and we will probably write another paper for the future version. 
Finally, the code is pretty messy right now and I will probably polish it the next version. **Most importantly, this is not representative of my software engineering skill, just in case you are considering hiring me.** 

### Environment

Run ```conda env create -f  d3_0514.yaml``` to set up the environment. Then replace the transformers package source file ```transformers/generation_utils.py``` with the ```generation_utils.py``` file with the one in this folder (where we implemented the ensemble_sampling). You can find where transformers is installed with the command: ```python3 -c "import transformers; print(transformers.__file__)"```

### Hardware

We ran the system with a single A100 with 80GB memory. If you are working with smaller GPUs, you might need to write a little additional code to use model parallelism to run the T5 models.

### Example Usage

Run ```python3 end2end_d3.py``` to describe the differences between pairs of text distributions from our benchmark. The core function is ```describe```, with the signature:

```describe(pos: List[str], neg: List[str]) -> Dict[str, float]```

It takes in two lists of strings corresponding to the samples from each distribution and returns a dictionary that maps each description to our score that approximates how well it can describe the differences. 

### Example Predictions

Run ```python3 examine_predictions_on_benchmark.py``` to examine example predictions on our benchmark. Change the value of the ```TOP_K``` variable to vary the number of descriptions you want to examine per distribution pairs.  

### Sketch of the System

Our system has 3 parts. 1) scoring samples according to how well it represents the difference between the two distributions, 2) propose hypotheses, and 3) verify hypotheses.

#### Part 1

Representative samples: implemented in ```get_extreme_w_highlight.py``` . The implementation is mostly consistent with that of the paper, though we also added an attention layer to highlight "salient" words.

#### Part 2

Proposer: implemented in ```proposer_wrapper.py``` . We provide two options a) using GPT-3 --- unfortunately we cannot directly share our fine-tuned proposer with you, so you need to fine-tune GPT-3 on your own using OpenAI's API. To make it more convenient for other researchers, we also implemented b) a proposer based on T5; this model is fine-tuned on our task-specific dataset and the instruction following dataset from [AI2 natural instruction v2.0 dataset](https://instructions.apps.allenai.org), and we also implemented an approach that ensembles the logits that uses input prompts with different samples. 

Finally, we noticed that we made some stupid errors that added noise to the fine-tuning data, and we fixed them when fine-tuning this version (so the performance could have been higher in our paper). 

We thank Dong Yang for fine-tuning the proposer. 

#### Part 3

Verifier: implemented in ```verifier_wrapper.py``` . The implementation is mostly consistent with the one describe in our paper. Again, we noticed that we made some stupid errors that added noise to the fine-tuning data, and we fixed them when fine-tuning this version.
