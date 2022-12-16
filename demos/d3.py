import os
import random
import pickle as pkl
import iprompt
import json
from typing import List
import tqdm
import imodelsx

if __name__ == '__main__':
    # distribution_pairs = json.load(open('benchmark.json'))

    positive_samples = [
        "How much in miles is a ten K run?",
        "When is the Jimmy Buffett concert coming to the E center in Camden NJ?",
        "What chapter of Gone with the Wind has Rhett Butler leaving Scarlett O 'Hara?",
        "What is the latitude and longitude of El Paso, Texas?",
        "How old was Elvis Presley when he died?"
    ]

    negative_samples = [
        "What is the daily requirement of folic acid for an expectant mother?",
        "What type of bridge is the Golden Gate Bridge?",
        "Where do the Blackhawks maintain their operations?",
        "What attorneys work for The Center for the Defense of Free Enterprise?",
        "What college football team did Knute Rockne build into a power?"
    ]

    hypotheses, hypothesis_scores = imodelsx.explain_datasets_d3(
        pos=positive_samples, # List[str] of positive examples
        neg=negative_samples, # another List[str]
        num_steps=100,
        num_folds=2,
        batch_size=64,
    )

    print('learned hypotheses', hypotheses)
    print('corresponding scores', hypothesis_scores)

    pkl.dump({'hypotheses': hypotheses, 'hypothesis_scores': hypothesis_scores},
             open('example_results.pkl', 'wb'))
