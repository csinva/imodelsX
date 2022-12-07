from transformers import BertModel, DistilBertModel
from transformers import AutoModelForCausalLM
from os.path import join as oj
import torch

def generate_ngrams_list(
    sentence: str,
    ngrams: int,
    tokenizer_ngrams,
    all_ngrams=False,
    parsing: str='',
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
        unigrams_list = [x for x in tokenizer_ngrams(sentence)]
        if all_ngrams:
            ngram_lengths = range(1, ngrams + 1)
    #         seqs = [str(x) for x in simple_tokenizer(sentence)] # precompute length 1
        else:
            ngram_lengths = range(ngrams, ngrams + 1)

        for ngram_length in ngram_lengths:
            for idx_starting in range(0, len(unigrams_list) + 1 - ngram_length):
                idx_ending = idx_starting + ngram_length
                seq = ''.join([t.text + t.whitespace_
                               for t in unigrams_list[idx_starting: idx_ending]]).strip()  # convert the tokens back to text
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


def get_model(checkpoint):
    if 'distilbert' in checkpoint.lower():
        model = DistilBertModel.from_pretrained(checkpoint)
    elif 'bert-base' in checkpoint.lower() or 'BERT' in checkpoint:
        model = BertModel.from_pretrained(checkpoint)
    elif 'gpt' in checkpoint.lower():
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, output_hidden_states=True)
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
    padded_tokens = [tok_ids + [0] *
                     (max_prompt_len - len(tok_ids)) for tok_ids in token_ids]
    input_ids = torch.LongTensor(padded_tokens)
    attn_mask = torch.zeros(input_ids.shape).long()
    for ix, tok_ids in enumerate(token_ids):
        attn_mask[ix][:len(tok_ids)] = 1

    # tokens = tokenizer(seqs, truncation=True, return_tensors="pt")
    return {'input_ids': input_ids, 'attention_mask': attn_mask}


def embed_and_sum_function(
    example,
    model,
    ngrams: int,
    tokenizer_embeddings,
    tokenizer_ngrams,
    checkpoint: str,
    dataset_key_text: str = None,
    layer: str = 'last_hidden_state',
    padding: bool = True,
    parsing: str = '',
    nlp_chunks = None,
    all_ngrams: bool=False,
):
    """Get summed embeddings for a single example
    Note: this function gets called many times, so don't want to do things like load a model here


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
    """
    if dataset_key_text is not None:
        sentence = example[dataset_key_text]
    else:
        sentence = example
    # seqs = sentence

    assert isinstance(sentence, str), 'sentence must be a string (batched mode not supported)'
    seqs = generate_ngrams_list(
        sentence, ngrams=ngrams, tokenizer_ngrams=tokenizer_ngrams,
        parsing=parsing, nlp_chunks=nlp_chunks, all_ngrams=all_ngrams,
    )
    # seqs = list(map(generate_ngrams_list, sentence))


    seq_len = len(seqs)
    if seq_len == 0:
        seqs = ["dummy"] # will multiply embedding by 0 so doesn't matter

    if 'bert' in checkpoint.lower():  # has up to two keys, 'last_hidden_state', 'pooler_output'
        if not hasattr(tokenizer_embeddings, 'pad_token') or tokenizer_embeddings.pad_token is None:
            tokenizer_embeddings.pad_token = tokenizer_embeddings.eos_token
        tokens = tokenizer_embeddings(seqs, padding=padding,
                                      truncation=True, return_tensors="pt")
        tokens = tokens.to(model.device)
        output = model(**tokens)
        if layer == 'pooler_output':
            embs = output['pooler_output'].cpu().detach().numpy()
        elif layer == 'last_hidden_state_mean' or layer == 'last_hidden_state':
            embs = output['last_hidden_state'].cpu().detach().numpy()
            embs = embs.mean(axis=1)
    elif 'gpt' in checkpoint.lower():
        tokens = preprocess_gpt_token_batch(seqs, tokenizer_embeddings)
        tokens = tokens.to(model.device)
        output = model(**tokens)

        # tuple of (layer x (batch_size, seq_len, hidden_size))
        h = output['hidden_states']
        # (batch_size, seq_len, hidden_size)
        embs = h[0].cpu().detach().numpy()
        embs = embs.mean(axis=1)  # (batch_size, hidden_size)

    # sum over the embeddings
    embs = embs.sum(axis=0).reshape(1, -1)
    if seq_len == 0:
        embs *= 0

    return {'embs': embs, 'seq_len': len(seqs)}
