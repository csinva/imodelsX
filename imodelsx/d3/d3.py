import os
import random
import pickle as pkl
import torch
import numpy as np

from typing import Tuple
from imodelsx.d3.step1_get_extreme import return_extreme_values
from imodelsx.d3.step2_proposer import init_proposer
from imodelsx.d3.step3_verifier import init_verifier
import json
from typing import List, Dict
import tqdm

"""
Describing Differences between Text Distributions with Natural Language

Ruiqi Zhong, Charlie Snell, Dan Klein, Jacob Steinhardt
https://arxiv.org/abs/2201.12323
"""


def explain_dataset_d3(
    pos: List[str],
    neg: List[str],
    proposer_name: str = 't5ruiqi-zhong/t5proposer_0514',
    verifier_name: str = 'ruiqi-zhong/t5verifier_0514',
    save_folder: str='results',
    num_steps: int=500,
    num_folds: int=2,    # default 4
    batch_size: int=32,  # default 16
    verbose: bool=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """This function returns hypothesis that describe the difference between
    the distributions present in two lists of strings

    Parameters
    ----------
    pos : List[str]
        a list of text samples from D_1
    neg : List[str]
        a list of text samples from D_0
    proposer_name : str, optional
        the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "t5t5-small" to debug, by default 't5ruiqi-zhong/t5proposer_0514'
    verifier_name : str, optional
        the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug, by default 'ruiqi-zhong/t5verifier_0514'
    save_folder : str, optional
        the folder to save the results, by default 'results'
    num_steps : int, optional
        the number of steps to run the algorithm, by default 500
    num_folds : int, optional
        the number of folds to use in cross-validation, by default 2
    batch_size : int, optional
    verbose : bool, optional
        whether to print intermediate updates, by default True

    Returns
    -------
    hypotheses: np.ndarray[str]
        String hypotheses for differentiating the classes
    hypothesis_scores: np.ndarray[float]
        Score for each hypothesis (how well it matches the positive class - the negative class)
    """
    # saving the initial arguments
    if save_folder is None:
        save_folder = 'end2end_jobs/' + str(random.random())
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        if verbose:
            print('Folder %s exists' % save_folder)
    if verbose:
        print('results will be saved to %s' % save_folder)

    # get samples that are representative of the differences between two distributions
    if verbose:
        print('\n**************************\nStep 1/3: get extreme samples...')
    extreme_vals = return_extreme_values(
        pos, neg, num_steps, num_folds, batch_size)
    pkl.dump(extreme_vals, open(os.path.join(
        save_folder, '01_extreme_vals.pkl'), 'wb'))

    with torch.no_grad():
        # propose hypotheses
        # Warning: proposer.inference_on_ensemble_prompts is currently not using ensembling!
        if verbose:
            print('\nStep 2/3: propose hypothesis...')
        pos2score, neg2score = extreme_vals['pos2score'], extreme_vals['neg2score']
        proposer = init_proposer(proposer_name)
        proposed_hypotheses = proposer.propose_hypothesis(pos2score, neg2score)
        pkl.dump(proposed_hypotheses, open(os.path.join(
            save_folder, '02_proposed_hypotheses.pkl'), 'wb'))

        # verify the hypotheses
        if verbose:
            print('\nStep 3/3: very hypotheses...')
        verifier = init_verifier(verifier_name)
        h2result = {}
        for h in set(proposed_hypotheses):
            h2result[h] = verifier.return_verification(h, pos, neg, 500)
        pkl.dump(h2result, open(os.path.join(
            save_folder, '03_verified_hypotheses.pkl'), 'wb'))
        # note: h_score is pos_score - neg_score for verifier
        h_scores = {h: v['h_score'] for h, v in h2result.items()}
        hypotheses = np.array(list(h_scores.keys()))
        hypothesis_scores = np.array(list(h_scores.values()))
        args_sorted = np.argsort(hypothesis_scores)[::-1]
        return hypotheses[args_sorted], hypothesis_scores[args_sorted]

