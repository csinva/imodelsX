from transformers import BertModel, BertConfig
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import numpy as np
import pickle as pkl
import os
from os.path import join as oj

def generate_decomposed_ngrams(sentence, ngrams=1):
    """seq of sequences to input (can do this better - stemming, etc.)
    """
    seqs = []
    unigrams_list = sentence.split(' ')
    for ngram_length in range(1, ngrams + 1):
        for idx_starting_word in range(0, len(unigrams_list) + 1 - ngram_length):
            if ngram_length == 1:
                seqs.append(unigrams_list[idx_starting_word])
            else:
                seqs.append(' '.join(
                    unigrams_list[idx_starting_word: idx_starting_word + ngram_length]))
    return seqs

def embed_and_sum_function(example):
    sentence = example['sentence']
    # seqs = sentence

    if isinstance(sentence, str):
        seqs = generate_decomposed_ngrams(sentence)
    elif isinstance(sentence, list):
        raise Exception('batched mode not supported')
        # seqs = list(map(generate_decomposed_ngrams, sentence))
    # print('seqs', type(seqs), seqs)
    
                            
    # maybe a smarter way to deal with pooling here?
    tokens = tokenizer(seqs, padding=True, truncation=True, return_tensors="pt")
    # print('tokens', tokens['input_ids'].shape, tokens['input_ids'])
    output = model(**tokens) # has two keys, 'last_hidden_state', 'pooler_output'
    embs = output['pooler_output'].cpu().detach().numpy()
    # print('embs', embs.shape)
    
    # sum over the embeddings
    embs = embs.sum(axis=0).reshape(1, -1)
    # print('embs', embs.shape)    
    
    return {'embs': embs}

if __name__ == '__main__':
    
    
    
    # set up model
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = BertModel.from_pretrained(checkpoint)
    
    
    # set up data
    dataset = datasets.load_dataset('sst2')

    # run 
    embedded_dataset = dataset.map(embed_and_sum_function) #, batched=True)
    
    # save
    save_dir = "data/processed/ngram=1_" + checkpoint
    os.makedirs(save_dir, exist_ok=True)
    embedded_dataset.save_to_disk(save_dir)
    for k in ['train', 'validation', 'test']:
        embs = np.array(dataset[k]['embs']).squeeze()
        pkl.dump(embs, open(oj(save_dir, 'embs_' + k + '.pkl'), 'wb'))