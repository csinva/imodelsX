import os
import random
import pickle as pkl
from get_extreme import return_extreme_values
from proposer import init_proposer
from verifier import init_verifier
import json
from typing import List
import tqdm

def describe(
    pos: List[str], # a list of text samples from D_1
    neg: List[str], # a list of text samples from D_0
    note: str='', # a note about this distribution, for logging purposes
    proposer_name: str='t5ruiqi-zhong/t5proposer_0514', # the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "t5t5-small" to debug
    verifier_name: str='ruiqi-zhong/t5verifier_0514', # the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug
    save_folder='results',
    num_steps=500, # default 2000
    num_folds=2,    # default 4
    batch_size=32,  # default 16
):
    # saving the initial arguments
    if save_folder is None:
        save_folder = 'end2end_jobs/' + str(random.random())
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print('Folder %s exists' % save_folder)
    print('results will be saved to %s' % save_folder)
    spec = {
        'note': note,
        'pos': pos,
        'neg': neg,
        'proposer_name': proposer_name,
        'verifier_name': verifier_name 
    }
    for k in ['note', 'proposer_name', 'verifier_name']:
        print(k, spec[k])
    pkl.dump(
        spec, open(os.path.join(save_folder, 'spec.pkl'), 'wb')
    )
    
    # get samples that are representative of the differences between two distributions
    extreme_vals = return_extreme_values(pos, neg, num_steps, num_folds, batch_size)
    pkl.dump(extreme_vals, open(os.path.join(save_folder, 'get_extreme_result.pkl'), 'wb'))
    
    # propose hypotheses
    pos2score, neg2score = extreme_vals['pos2score'], extreme_vals['neg2score']
    proposer = init_proposer(proposer_name)
    proposed_hypotheses = proposer.propose_hypothesis(pos2score, neg2score)
    
    pkl.dump(proposed_hypotheses, open(os.path.join(save_folder, 'proposed_hypotheses.pkl'), 'wb'))
    
    # verify the hypotheses
    verifier = init_verifier(verifier_name)
    h2result = {}
    for h in set(proposed_hypotheses):
        h2result[h] = verifier.return_verification(h, pos, neg, 500)
    
    pkl.dump(h2result, open(os.path.join(save_folder, 'scored_hypotheses.pkl'), 'wb'))
    return {h: v['h_score'] for h, v in h2result.items()}


if __name__ == '__main__':
    distribution_pairs = json.load(open('benchmark_sec_4/benchmark.json'))

    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):
        h2score = describe(pos=d['positive_samples'], 
                           neg=d['negative_samples'], 
                           note='benchmark %d; can be anything, for logging purpose only' % i)
        all_h2score.append(h2score)
        pkl.dump(all_h2score, open('benchmark_h2score.pkl', 'wb'))