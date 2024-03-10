from os.path import join as oj
from datasets import Dataset
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import imodelsx.util
from typing import List


def embed_and_sum_function(
    example,
    model,
    ngrams: int,
    tokenizer_embeddings,
    tokenizer_ngrams,
    checkpoint: str,
    dataset_key_text: str = None,
    layer: str = "last_hidden_state",
    padding: str = True,
    batch_size: int = 8,
    parsing: str = "",
    nlp_chunks=None,
    all_ngrams: bool = False,
    fit_with_ngram_decomposition: bool = True,
    embedding_prefix: str = "Represent the short phrase for sentiment classification: ",
    embedding_suffix: str = "",
    embedding_strategy: str = 'mean',
    sum_embeddings=True,
    prune_stopwords=False,
):
    """Get summed embeddings for a single example

    Params
    ------
    ngrams: int
        What order of ngrams to use (1 for unigrams, 2 for bigrams, ...)
    dataset_key_text:
        str that identifies where data examples are stored, e.g. "sentence" for sst2
    tokenizer_embeddings
        tokenizing for the embedding model
    tokenizer_ngrams
        tokenizing the ngrams (word-based tokenization is more interpretable)
    layer: str
        which layer to extract embeddings from
    batch_size: int
        batch size for simultaneously running ngrams (for a single example)
    parsing: str
        whether to use parsing rather than extracting all ngrams
    nlp_chunks
        if parsing is not empty string, a parser that extracts specific ngrams
    fit_with_ngram_decomposition
        whether to fit the model with ngram decomposition (if not just use the standard sentence)
    embedding_prefix
        if checkpoint is an instructor/autoregressive model, prepend this prompt
    embedding_suffix
        if checkpoint is an autoregressive model, append this prompt
    embedding_strategy: str
        'mean': compute mean over ngram tokens
        'next_token_distr': use next token distribution as an embedding (requires AutoModelForCausalLM checkpoint)
    all_ngrams: bool
        whether to include all ngrams of lower order
    """

    # convert to list of strings
    seqs = _get_seqs(
        example, dataset_key_text, fit_with_ngram_decomposition,
        ngrams, tokenizer_ngrams, parsing, nlp_chunks, all_ngrams, prune_stopwords)
    if embedding_strategy == 'next_token_distr':
        seqs = [f'{embedding_prefix}{x_i}{embedding_suffix}' for x_i in seqs]

    if not checkpoint.startswith("hkunlp/instructor") and (
        not hasattr(tokenizer_embeddings, "pad_token")
        or tokenizer_embeddings.pad_token is None
    ):
        tokenizer_embeddings.pad_token = tokenizer_embeddings.eos_token

    # compute embeddings
    embs = []
    if checkpoint.startswith("hkunlp/instructor"):
        embs = model.encode(
            [[embedding_prefix, x_i] for x_i in seqs], batch_size=batch_size
        )
    else:
        tokens = tokenizer_embeddings(
            seqs, padding=padding, truncation=True, return_tensors="pt"
        )

        ds = Dataset.from_dict(tokens).with_format("torch")

        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                output = model(**batch)
            torch.cuda.empty_cache()
            if embedding_strategy == "next_token_distr":
                emb = _next_token_distr_with_mask(
                    output["logits"], batch["attention_mask"]
                )
            else:
                if layer == "pooler_output":
                    emb = output["pooler_output"]
                elif layer == "last_hidden_state_mean" or layer == "last_hidden_state":
                    # extract (batch_size, seq_len, hidden_size)
                    emb = output["last_hidden_state"]

                    # convert to (batch_size, hidden_size)
                    emb = _mean_with_mask(emb, batch["attention_mask"])

                elif "hidden_states" in output.keys():
                    # extract (layer x (batch_size, seq_len, hidden_size))
                    h = output["hidden_states"]

                    # convert to (batch_size, seq_len, hidden_size)
                    emb = h[0]

                    # convert to (batch_size, hidden_size)
                    emb = _mean_with_mask(emb, batch["attention_mask"])
                else:
                    raise Exception(f"keys: {output.keys()}")

            embs.append(emb.cpu().detach().numpy())

        embs = np.concatenate(embs)

    # else:
        # raise Exception(f"Unknown model checkpoint {checkpoint}")

    # sum over the embeddings
    if sum_embeddings:
        embs = embs.sum(axis=0).reshape(1, -1)
    if len(seqs) == 0:
        embs *= 0

    return {"embs": embs, "seq_len": len(seqs)}


def _mean_with_mask(emb_batch, mask_batch):
    '''Compute the mean of embeddings ignoring masked tokens
    '''
    # create a mask for real tokens
    expanded_attention_mask = mask_batch.unsqueeze(
        -1).expand_as(emb_batch)

    # compute the sum of embeddings for real tokens and count the real tokens
    sum_embeddings = (emb_batch * expanded_attention_mask).sum(1)
    real_token_counts = expanded_attention_mask.sum(1)

    # avoid division by zero for completely padded sequences by setting count to 1 where it's 0
    real_token_counts = real_token_counts.masked_fill_(
        real_token_counts == 0, 1)

    return sum_embeddings / real_token_counts


def _next_token_distr_with_mask(logits_batch, mask_batch):
    '''Return the logits of the first non-masked token
    '''
    # get the real token counts
    real_token_counts = mask_batch.sum(1) - 1

    # get the logits of the next token
    next_token_logits = logits_batch[
        range(len(logits_batch)), real_token_counts]

    return next_token_logits


def _get_seqs(
        example, dataset_key_text, fit_with_ngram_decomposition,
        ngrams, tokenizer_ngrams, parsing, nlp_chunks, all_ngrams, prune_stopwords) -> List[str]:

    if dataset_key_text is not None:
        sentence = example[dataset_key_text]
    else:
        sentence = example

    if fit_with_ngram_decomposition:
        seqs = imodelsx.util.generate_ngrams_list(
            sentence,
            ngrams=ngrams,
            tokenizer_ngrams=tokenizer_ngrams,
            parsing=parsing,
            nlp_chunks=nlp_chunks,
            all_ngrams=all_ngrams,
            prune_stopwords=prune_stopwords,
        )
    elif isinstance(sentence, list):
        seqs = sentence
    elif isinstance(sentence, str):
        seqs = [sentence]
    else:
        raise ValueError("sentence must be a string or list of strings")

    seq_len = len(seqs)
    if seq_len == 0:
        # will multiply embedding by 0 so doesn't matter, but still want to get the shape
        seqs = ["dummy"]
    return seqs


def _clean_np_array(arr):
    """Replace inf and nan with 0"""
    arr[np.isinf(arr)] = 0
    arr[np.isnan(arr)] = 0
    return arr
