

import logging
from typing import List
from tqdm import tqdm
from transformers import pipeline
import datasets
import numpy as np
from collections import Counter


def generate_ngrams_list(
    sentence: str,
    ngrams: int,
    tokenizer_ngrams=None,
    all_ngrams=False,
    parsing: str = '',
    nlp_chunks=None,
    pad_starting_ngrams=False,
    pad_ending_ngrams=False,
    min_frequency=1,
):
    """Get list of ngrams from sentence using a tokenizer

    Params
    ------
    ngrams: int
        What order of ngrams to use (1 for unigrams, 2 for bigrams, ...)
    all_ngrams: bool
        whether to include all n-grams up to n or just n
    pad_starting_ngrams: bool
        if all_ngrams=False, then pad starting ngrams with shorter length ngrams
        so that length of ngrams_list is the same as the initial sequence
        e.g. for ngrams=3 ["the", "the quick", "the quick brown", "quick brown fox", "brown fox jumps", ...]
    pad_ending_ngrams: bool
    min_frequency: int
        minimum frequency to be considered for the ngrams_list
    """

    seqs = []

    if tokenizer_ngrams is None:
        tokenizer_ngrams = lambda x: x.split()

    # unigrams
    unigrams_list = [str(x) for x in tokenizer_ngrams(sentence)]
    if ngrams == 1:
        seqs += unigrams_list

    # all ngrams in loop
    else:
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

    if pad_starting_ngrams:
        assert all_ngrams is False, "pad_starting_ngrams only works when all_ngrams=False"
        seqs_init = [' '.join(unigrams_list[:ngram_length]) for ngram_length in range(1, ngrams)]
        seqs = seqs_init + seqs

    if pad_ending_ngrams:
        assert all_ngrams is False, "pad_ending_ngrams only works when all_ngrams=False"
        seqs_end = [' '.join(unigrams_list[-ngram_length:]) for ngram_length in range(1, ngrams)][::-1]
        seqs = seqs + seqs_end
    
    freqs = Counter(seqs)

    seqs = [seq for seq, freq in freqs.items() if freq >= min_frequency]

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


def get_spacy_tokenizer(convert_output=True, convert_lower=True):
    from spacy.lang.en import English
    nlp = English()
    if convert_output:
        class LLMTreeTokenizer:
            def __init__(self):
                self.tok = nlp

            # written kind of weirdly to optimize the speed of the tokenizer
            if convert_lower:
                def __call__(self, s):
                    s = s.lower()
                    return [str(x) for x in self.tok(s)]
            else:
                def __call__(self, s):
                    return [str(x) for x in self.tok(s)]
        return LLMTreeTokenizer()
    else:
        return nlp


STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'nor', 'only', 'own', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
