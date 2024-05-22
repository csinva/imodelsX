'''This is a simplified example, many optimizations for performance can be made (e.g. using sglang)
'''
import numpy as np
from typing import List
from os.path import expanduser
from tqdm import tqdm
import imodelsx.llm
import pandas as pd
import warnings


class QAEmb:
    def __init__(
            self,
            questions: List[str],
            checkpoint: str = 'mistralai/Mistral-7B-Instruct-v0.2',
            use_cache: bool = True,
            batch_size: int = 16
    ):
        checkpoints_tested = [
            'gpt2',
            'gpt2-xl',
            'mistralai/Mistral-7B-Instruct-v0.2',
            'meta-llama/Meta-Llama-3-8B-Instruct',
            'meta-llama/Meta-Llama-3-8B-Instruct-fewshot',
            'meta-llama/Meta-Llama-3-8B-Instruct-refined',
        ]
        if not checkpoint in checkpoints_tested:
            warnings.warn(
                f"Checkpoint {checkpoint} has not been tested. You may want to check that everything is running smoothly.")
        self.questions = questions
        if 'mistral' in checkpoint and 'Instruct' in checkpoint:
            self.prompt = "<s>[INST]'Input text: {example}\nQuestion: {question}\nAnswer with yes or no, then give an explanation.[/INST]"
            self.checkpoint = checkpoint
        elif 'Meta-Llama-3' in checkpoint and 'Instruct' in checkpoint:
            if '-refined' in checkpoint:
                self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRead the input then answer a question about the input.\n**Input**: "{example}"\n**Question**: {question}\nAnswer with yes or no, then give an explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n**Answer**:'
                self.checkpoint = checkpoint.replace('-refined', '')
            elif '-fewshot' in checkpoint:
                self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a concise, helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: and i just kept on laughing because it was so\nQuestion: Does the input mention laughter?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nYes<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: what a crazy day things just kept on happening\nQuestion: Is the sentence related to food preparation?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNo<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: i felt like a fly on the wall just waiting for\nQuestion: Does the text use a metaphor or figurative language?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nYes<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: he takes too long in there getting the pans from\nQuestion: Is there a reference to sports?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNo<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: was silent and lovely and there was no sound except\nQuestion: Is the sentence expressing confusion or uncertainty?\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nNo<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: {example}\nQuestion: {question}\nAnswer with Yes or No.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                self.checkpoint = checkpoint.replace('-fewshot', '')
            else:
                self.prompt = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a concise, helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput text: {example}\nQuestion: {question}\nAnswer with yes or no, then give an explanation.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
                self.checkpoint = checkpoint
        else:
            self.prompt = 'Input: {example}\nQuestion: {question} Answer yes or no.\nAnswer:'
            self.checkpoint = checkpoint

        self.llm = imodelsx.llm.get_llm(
            self.checkpoint, CACHE_DIR=expanduser("~/cache_qa_embedder"))
        self.batch_size = batch_size
        self.use_cache = use_cache

    def __call__(self, examples: List[str], verbose=True) -> np.ndarray:
        '''
        Returns
        -------
        embeddings: (num_examples, num_questions)
        '''
        programs = [
            self.prompt.format(example=example, question=question)
            for example in examples
            for question in self.questions
        ]

        # run in batches
        answers = []
        # pipeline uses batch_size under the hood, but use for-loop here to get progress bar
        batch_size_mult = self.batch_size * 8
        for i in tqdm(range(0, len(programs), batch_size_mult)):
            answers += self.llm(
                programs[i:i+batch_size_mult],
                max_new_tokens=1,
                verbose=verbose,
                use_cache=self.use_cache,
                batch_size=self.batch_size,
            )

        answers = list(map(lambda x: 'yes' in x.lower(), answers))
        answers = np.array(answers).reshape(len(examples), len(self.questions))
        embeddings = np.array(answers, dtype=float)

        return embeddings