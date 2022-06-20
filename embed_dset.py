from transformers import BertModel, BertConfig
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import numpy as np
import pickle as pkl
import os
from os.path import join as oj
from spacy.lang.en import English

def generate_decomposed_ngrams(sentence):
    """seq of sequences to input (can do this better - stemming, etc.)
    
    Params
    ------
    ngrams: int
    """
    unigrams_list = [str(x) for x in simple_tokenizer(sentence)]
    # unigrams_list = sentence.split(' ')
    if ngrams == 1:
        return unigrams_list
    seqs = unigrams_list.copy()
    for ngram_length in range(2, ngrams + 1):
        for idx_starting_word in range(0, len(unigrams_list) + 1 - ngram_length):
                seqs.append(' '.join(
                    unigrams_list[idx_starting_word: idx_starting_word + ngram_length]))
    # print('seqs', seqs)
    return seqs

def embed_and_sum_function(example):
    """
    Params
    ------
    padding: True, "max_length"
    """
    sentence = example['sentence']
    # seqs = sentence

    if isinstance(sentence, str):
        seqs = generate_decomposed_ngrams(sentence)
    elif isinstance(sentence, list):
        raise Exception('batched mode not supported')
        # seqs = list(map(generate_decomposed_ngrams, sentence))
    # print('seqs', type(seqs), seqs)
    
                            
    # maybe a smarter way to deal with pooling here?
    tokens = tokenizer(seqs, padding=padding, truncation=True, return_tensors="pt")
    # print('tokens', tokens['input_ids'].shape, tokens['input_ids'])
    output = model(**tokens) # has two keys, 'last_hidden_state', 'pooler_output'
    embs = output['pooler_output'].cpu().detach().numpy()
    # print('embs', embs.shape)
    
    # sum over the embeddings
    embs = embs.sum(axis=0).reshape(1, -1)
    # print('embs', embs.shape)    
    
    return {'embs': embs}

if __name__ == '__main__':
    
    # hyperparams
    padding = 'max_length' # True
    checkpoint = "bert-base-uncased"
    ngrams = 1
    
    
    # set up model
    nlp = English()
    simple_tokenizer = nlp.tokenizer # for our word-finding
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) # for actually passing things to the model
    model = BertModel.from_pretrained(checkpoint)
    
    
    # set up data
    dataset = datasets.load_dataset('sst2')

    # run 
    embedded_dataset = dataset.map(embed_and_sum_function) #, batched=True)
    
    # save
    save_dir = f"data/processed/ngram={ngrams}_" + checkpoint + "_" + padding
    os.makedirs(save_dir, exist_ok=True)
    embedded_dataset.save_to_disk(save_dir)
    for k in ['train', 'validation', 'test']:
        embs = np.array(dataset[k]['embs']).squeeze()
        pkl.dump(embs, open(oj(save_dir, 'embs_' + k + '.pkl'), 'wb'))