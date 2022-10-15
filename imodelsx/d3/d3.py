import os
import random
import pickle as pkl
import torch

from traitlets import Dict
from imodelsx.d3.get_extreme import return_extreme_values
from imodelsx.d3.proposer import init_proposer
from imodelsx.d3.verifier import init_verifier
import json
from typing import List
import tqdm


def explain_datasets_d3(
    pos: List[str],  # a list of text samples from D_1
    neg: List[str],  # a list of text samples from D_0
    note: str = '',  # a note about this distribution, for logging purposes
    # the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "t5t5-small" to debug
    proposer_name: str = 't5ruiqi-zhong/t5proposer_0514',
    # the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug
    verifier_name: str = 'ruiqi-zhong/t5verifier_0514',
    save_folder: str='results',
    num_steps: int=500,  # default 2000
    num_folds: int=2,    # default 4
    batch_size: int=32,  # default 16
    verbose: bool=True,
) -> Dict:
    """
    Warning: proposer.inference_on_ensemble_prompts is currently not using ensembling!
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
    save_args(args=locals(), verbose=verbose)

    # get samples that are representative of the differences between two distributions
    if verbose:
        print('\nStep 1/3: get extreme samples...')
    extreme_vals = return_extreme_values(
        pos, neg, num_steps, num_folds, batch_size)
    pkl.dump(extreme_vals, open(os.path.join(
        save_folder, '01_extreme_vals.pkl'), 'wb'))

    with torch.no_grad():
        # propose hypotheses
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
        return {h: v['h_score'] for h, v in h2result.items()}


def save_args(args: Dict, verbose=True):
    if verbose:
        for k in ['note', 'proposer_name', 'verifier_name']:
            print(k, args[k])
    pkl.dump(
        args, open(os.path.join(args['save_folder'], 'args.pkl'), 'wb')
    )
