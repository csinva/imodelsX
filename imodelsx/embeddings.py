import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import imodelsx.embeddings
from copy import deepcopy


def get_embs(
    texts: List[str], checkpoint: str = "bert-base-uncased", batch_size: int = 32,
    aggregate: str = "mean"
) -> np.ndarray:
    '''
    Get embeddings for a list of texts.

    Params
    ------
    texts: List[str]: List of texts to get embeddings for.
    checkpoint: str: Name of the checkpoint to use. Use tf-idf for linear embeddings.
    batch_size: int: Batch size to use for inference.
    aggregate: str: Aggregation method to use for the embeddings. Can be "mean" or "first" (to use CLS token for BERT).
    '''
    if checkpoint == "tf-idf":
        return get_embs_linear(texts)

    # load model
    # get embeddings for each text from the corpus
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to("cuda")

    # calculate embeddings
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        t = texts[i: i + batch_size]
        with torch.no_grad():
            # tokenize
            inputs = tokenizer(
                t, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            # Shape: [batch_size, seq_len, hidden_size]
            outputs = model(**inputs).last_hidden_state.detach().cpu().numpy()
            # average over sequence length
            if aggregate == "mean":
                emb = np.mean(outputs, axis=1).squeeze()
            elif aggregate == "first":
                emb = outputs[:, 0, :].squeeze()  # use CLS token
            embs.append(deepcopy(emb))
    embs = np.concatenate(embs)
    return embs


def get_embs_linear(texts: List[str]) -> np.ndarray:
    """Get TF-IDF vectors for a list of texts.

    Parameters
    ----------
    texts (List[str]): List of texts to get TF-IDF vectors for.

    Returns
    -------
    embs: np.ndarray: TF-IDF vectors for the input texts.
    """
    vectorizer = TfidfVectorizer(
        # tokenizer=AutoTokenizer.from_pretrained(checkpoint).tokenize,
        # preprocessor=lambda x: x,
        # token_pattern=None,
        lowercase=False,
        max_features=10000,
    )
    return vectorizer.fit_transform(texts).toarray()
