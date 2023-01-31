

import logging
from typing import List
from tqdm import tqdm
from transformers import pipeline
import datasets
import numpy as np


def get_embs_llm(X: List[str], checkpoint: str):
    """Return embeddings from HF model given checkpoint name
    (Fixed-size embedding by averaging over seq_len)
    """
    pipe = pipeline(
        "feature-extraction",
        model=checkpoint,
        truncation=True,
        device=0
    )

    def get_emb(x):
        return {'emb': pipe(x['text'])}
    text = datasets.Dataset.from_dict({'text': X})
    out_list = text.map(get_emb)['emb']
    # out_list is (batch_size, 1, (seq_len + 2), 768)

    # convert to np array by averaging over len (can't just convert the since seq lens vary)
    num_examples = len(out_list)
    dim_size = len(out_list[0][0][0])
    embs = np.zeros((num_examples, dim_size))
    logging.info('extract embs HF...')
    for i in tqdm(range(num_examples)):
        embs[i] = np.mean(out_list[i], axis=1)  # avg over seq_len dim
    return embs
