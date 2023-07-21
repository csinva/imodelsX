from transformers import BertModel, DistilBertModel
from transformers import AutoModelForCausalLM
from os.path import join as oj
from datasets import Dataset
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import imodelsx.util


def get_model(checkpoint):
    if "distilbert" in checkpoint.lower():
        model = DistilBertModel.from_pretrained(checkpoint)
    elif "bert-base" in checkpoint.lower() or "BERT" in checkpoint:
        model = BertModel.from_pretrained(checkpoint)
    elif "gpt" in checkpoint.lower():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, output_hidden_states=True
        )
    try:
        model = model.cuda()
    except:
        pass
    return model


def preprocess_gpt_token_batch(seqs, tokenizer_embeddings):
    """Preprocess token batch with token strings of different lengths
    Add attention mask here
    """
    # batch_size = len(seqs)

    token_ids = [tokenizer_embeddings.encode(s, add_special_tokens=False) for s in seqs]
    prompt_lengths = [len(s) for s in token_ids]
    max_prompt_len = max(prompt_lengths)

    # use 0 as padding id, shouldn't matter (snippet from here https://github.com/huggingface/transformers/issues/3021)
    padded_tokens = [
        tok_ids + [0] * (max_prompt_len - len(tok_ids)) for tok_ids in token_ids
    ]
    input_ids = torch.LongTensor(padded_tokens)
    attn_mask = torch.zeros(input_ids.shape).long()
    for ix, tok_ids in enumerate(token_ids):
        attn_mask[ix][: len(tok_ids)] = 1

    # tokens = tokenizer(seqs, truncation=True, return_tensors="pt")
    return {"input_ids": input_ids, "attention_mask": attn_mask}


def embed_and_sum_function(
    example,
    model,
    ngrams: int,
    tokenizer_embeddings,
    tokenizer_ngrams,
    checkpoint: str,
    dataset_key_text: str = None,
    layer: str = "last_hidden_state",
    padding: str = "max_length",
    batch_size: int = 8,
    parsing: str = "",
    nlp_chunks=None,
    all_ngrams: bool = False,
    fit_with_ngram_decomposition: bool = True,
    instructor_prompt: str = None,
    sum_embeddings=True,
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
    parsing: str
        whether to use parsing rather than extracting all ngrams
    nlp_chunks
        if parsing is not empty string, a parser that extracts specific ngrams
    fit_with_ngram_decomposition
        whether to fit the model with ngram decomposition (if not just use the standard sentence)
    instructor_prompt: str
        if using instructor, the prompt to use
    """
    if dataset_key_text is not None:
        sentence = example[dataset_key_text]
    else:
        sentence = example
    # seqs = sentence

    if fit_with_ngram_decomposition:
        seqs = imodelsx.util.generate_ngrams_list(
            sentence,
            ngrams=ngrams,
            tokenizer_ngrams=tokenizer_ngrams,
            parsing=parsing,
            nlp_chunks=nlp_chunks,
            all_ngrams=all_ngrams,
        )
    elif isinstance(sentence, list):
        seqs = sentence
    elif isinstance(sentence, str):
        seqs = [sentence]
    else:
        raise ValueError("sentence must be a string or list of strings")
    # assert isinstance(
    #     sentence, str
    # ), "sentence must be a string (batched mode not supported)"
    # seqs = list(map(imodelsx.util.generate_ngrams_list, sentence))

    seq_len = len(seqs)
    if seq_len == 0:
        # will multiply embedding by 0 so doesn't matter, but still want to get the shape
        seqs = ["dummy"]

    if not checkpoint.startswith("hkunlp/instructor") and (
        not hasattr(tokenizer_embeddings, "pad_token")
        or tokenizer_embeddings.pad_token is None
    ):
        tokenizer_embeddings.pad_token = tokenizer_embeddings.eos_token

    embs = []

    if (
        "bert" in checkpoint.lower()
    ):  # has up to two keys, 'last_hidden_state', 'pooler_output'
        tokens = tokenizer_embeddings(
            seqs, padding=padding, truncation=True, return_tensors="pt"
        )

        ds = Dataset.from_dict(tokens).with_format("torch")

        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                output = model(**batch)
            torch.cuda.empty_cache()

            if layer == "pooler_output":
                emb = output["pooler_output"].cpu().detach().numpy()
            elif layer == "last_hidden_state_mean" or layer == "last_hidden_state":
                emb = output["last_hidden_state"].cpu().detach().numpy()
                emb = emb.mean(axis=1)

            embs.append(emb)

        embs = np.concatenate(embs)

    elif "gpt" in checkpoint.lower() or "llama" in checkpoint.lower():
        tokens = preprocess_gpt_token_batch(seqs, tokenizer_embeddings)

        ds = Dataset.from_dict(tokens).with_format("torch")

        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                output = model(**batch)
            torch.cuda.empty_cache()

            if "hidden_states" in output.keys():
                # extract (layer x (batch_size, seq_len, hidden_size))
                h = output["hidden_states"]

                # convert to (batch_size, seq_len, hidden_size)
                emb = h[0].cpu().detach().numpy()

                # convert to (batch_size, hidden_size)
                emb = emb.mean(axis=1)
            elif "last_hidden_state" in output.keys():
                # extract (batch_size, seq_len, hidden_size)
                h = output["last_hidden_state"]

                # convert to np
                emb = h.cpu().detach().numpy()

                # convert to (batch_size, hidden_size)
                emb = emb.mean(axis=1)
            else:
                raise Exception(f"keys: {output.keys()}")

            embs.append(emb)

        embs = np.concatenate(embs)

    elif checkpoint.startswith("hkunlp/instructor"):
        if instructor_prompt is None:
            instructor_prompt = (
                "Represent the short phrase for sentiment classification: "
            )
        embs = model.encode(
            [[instructor_prompt, x_i] for x_i in seqs], batch_size=batch_size
        )

    else:
        raise Exception(f"Unknown model checkpoint {checkpoint}")

    # sum over the embeddings
    if sum_embeddings:
        embs = embs.sum(axis=0).reshape(1, -1)
    if seq_len == 0:
        embs *= 0

    return {"embs": embs, "seq_len": len(seqs)}
