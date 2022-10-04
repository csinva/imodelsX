import os
import random
import pickle as pkl
import iprompt
import json
from typing import List
import tqdm


if __name__ == '__main__':
    distribution_pairs = json.load(open('benchmark.json'))
    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):
        h2score = iprompt.explain_d3(
            pos=d['positive_samples'], 
            neg=d['negative_samples'], 
            note=f'benchmark {i}; can be anything, for logging purpose only',
            num_steps=100,
            num_folds=2,
            batch_size=64,
        )
        all_h2score.append(h2score)
        pkl.dump(all_h2score, open('benchmark_h2score.pkl', 'wb'))