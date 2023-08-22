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


def get_embs(
    texts: List[str], checkpoint: str = "bert-base-uncased", batch_size: int = 32
) -> np.ndarray:
    # load model
    # get embeddings for each text from the corpus
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).to("cuda")

    # calculate embeddings
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        t = texts[i : i + batch_size]
        with torch.no_grad():
            # tokenize
            inputs = tokenizer(
                t, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            # Shape: [batch_size, seq_len, hidden_size]
            outputs = model(**inputs).last_hidden_state.detach().cpu().numpy()
            emb = np.mean(outputs, axis=1).squeeze()  # average over sequence length
            # emb = outputs[:, 0, :].squeeze()  # use CLS token
            embs.append(emb)
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
        preprocessor=lambda x: x,
        token_pattern=None,
        lowercase=False,
        max_features=10000,
    )
    return vectorizer.fit_transform(texts).toarray()
