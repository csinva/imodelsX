

import logging
from typing import List
from tqdm import tqdm
from transformers import pipeline
import datasets
import numpy as np

def generate_ngrams_list(
    sentence: str,
    ngrams: int,
    tokenizer_ngrams,
    all_ngrams=False,
    parsing: str = '',
    nlp_chunks=None,
):
    """Get list of ngrams from sentence using a tokenizer

    Params
    ------
    ngrams: int
        What order of ngrams to use (1 for unigrams, 2 for bigrams, ...)
    all_ngrams: bool
        whether to include all n-grams up to n or just n
    parsing
    """

    seqs = []

    # unigrams
    if ngrams == 1:
        seqs += [str(x) for x in tokenizer_ngrams(sentence)]

    # all ngrams in loop
    else:
        unigrams_list = [str(x) for x in tokenizer_ngrams(sentence)]
        if all_ngrams:
            ngram_lengths = range(1, ngrams + 1)
    #         seqs = [str(x) for x in simple_tokenizer(sentence)] # precompute length 1
        else:
            ngram_lengths = range(ngrams, ngrams + 1)

        for ngram_length in ngram_lengths:
            for idx_starting in range(0, len(unigrams_list) + 1 - ngram_length):
                idx_ending = idx_starting + ngram_length
                seq = ' '.join(unigrams_list[idx_starting: idx_ending]).strip()
                # seq = ''.join([t.text + ' ' #t.whitespace_
                            #    for t in unigrams_list[idx_starting: idx_ending]]).strip()  # convert the tokens back to text
                if len(seq) > 0 and not seq.isspace():  # str is not just whitespace
                    seqs.append(seq)

    # add noun_chunks which at least have a space in them
    if parsing == 'noun_chunks':
        doc = nlp_chunks(sentence)
        seqs += [
            chunk.text for chunk in doc.noun_chunks
            if ' ' in chunk.text
        ]
    return seqs

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
