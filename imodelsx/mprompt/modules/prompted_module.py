import logging
from typing import List
import datasets
from transformers import pipeline
import numpy as np
from tqdm import tqdm
import sklearn.preprocessing
from spacy.lang.en import English
import imodelsx
import imodelsx.util
import pickle as pkl
from os.path import dirname, join
import os.path
import re
import mprompt.methods.llm
from langchain import PromptTemplate
from mprompt.data.data import TASKS
modules_dir = dirname(os.path.abspath(__file__))


class PromptedModule():

    def __init__(self, task_str: str = 'toy_animal', checkpoint='gpt-xl'):
        """
        Params
        ------
        """
        print(f'loading {checkpoint}...')
        self.llm = mprompt.methods.llm.get_llm(checkpoint)
        self._init_task(task_str)

    def _init_task(self, task_str: str):
        self.task_str = task_str
        self.task = TASKS[task_str]
        self.prompt_template = PromptTemplate(
            input_variables=['input'],
            template=self.task['template'],
        )

    def __call__(self, X: List[str]) -> np.ndarray:
        """Returns a scalar continuous response for each element of X
        """
        probs = np.zeros(len(X))
        for i, x in enumerate(tqdm(X)):
            prompt = self.prompt_template.format(input=x)
            probs[i] = self.llm._get_logit_for_target_token(
                prompt, target_token_str=self.task['target_token'])
        return probs
    
    def generate(self, X: List[str]) -> List[str]:
        """Returns a text generation for each value of a list of strings
        Only really used for testing
        """
        generations = []
        for i, x in enumerate(tqdm(X)):
            prompt = self.prompt_template.format(input=x)
            generations.append(repr(prompt) + ' -> ' + repr(self.llm(prompt)))
        return generations


if __name__ == '__main__':
    mod = PromptedModule(task_str='d3_0_irony')
    X = mod.get_relevant_data()
    print('X', X)
    print(X[0][:50])
    resp = mod(X[:3])
    print(resp)
