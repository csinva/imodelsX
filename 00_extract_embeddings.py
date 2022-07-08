from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import data
import numpy as np
import pickle as pkl
import os
import os.path
from os.path import join as oj
from spacy.lang.en import English
import argparse
import config
import torch
path_to_current_file = os.path.dirname(os.path.abspath(__file__))


def generate_ngrams_list(sentence, all_ngrams=False):
    """get list of grams
    
    Params
    ------
    args.ngrams: int
    all_ngrams: bool
        whether to include all n-grams up to n or just n
    """
    
    # unigrams_list = sentence.split(' ')
    if args.ngrams == 1:
        return [str(x) for x in simple_tokenizer(sentence)]
    
    seqs = []
    unigrams_list = [x for x in simple_tokenizer(sentence)]
    if all_ngrams:
        ngram_lengths = range(1, args.ngrams + 1)
#         seqs = [str(x) for x in simple_tokenizer(sentence)] # precompute length 1
    else:
        ngram_lengths = range(args.ngrams, args.ngrams + 1)
        
        
    for ngram_length in ngram_lengths:
        for idx_starting in range(0, len(unigrams_list) + 1 - ngram_length):
            idx_ending = idx_starting + ngram_length
            seq = ''.join([t.text + t.whitespace_
                                 for t in unigrams_list[idx_starting: idx_ending]]).strip() # convert the tokens back to text
            if len(seq) > 0 and not seq.isspace(): # str is not just whitespace
                seqs.append(seq)
    # print('seqs', seqs)
    return seqs

def embed_and_sum_function(example):
    """
    Params
    ------
    args.padding: True, "max_length"
    args.dataset_key_text: str, e.g. "sentence" for sst2
    """
    sentence = example[args.dataset_key_text]
    # seqs = sentence

    if isinstance(sentence, str):
        seqs = generate_ngrams_list(sentence)
    elif isinstance(sentence, list):
        raise Exception('batched mode not supported')
        # seqs = list(map(generate_ngrams_list, sentence))
    # print('sentence:', sentence)
    # print('seqs:', type(seqs), seqs)
    
                            
    # maybe a smarter way to deal with pooling here?
    seq_len = len(seqs)
    if seq_len == 0:
        seqs = ["dummy"]
        
    tokens = tokenizer(seqs, padding=args.padding, truncation=True, return_tensors="pt")
    # print('tokens', tokens['input_ids'].shape, tokens['input_ids'])
    output = model(**tokens) # has two keys, 'last_hidden_state', 'pooler_output'
    embs = output['pooler_output'].cpu().detach().numpy()
    # print('embs', embs.shape)
    
    # sum over the embeddings
    embs = embs.sum(axis=0).reshape(1, -1)
    if seq_len == 0:
        embs *= 0
    # print('embs', embs.shape)    
    
    return {'embs': embs, 'seq_len': len(seqs)}

def get_dir_name(args, ngrams=None):
    if not ngrams:
        ngrams = args.ngrams
    return f"ngram={ngrams}_" + 'sub=' + str(args.subsample) + '_' + args.checkpoint.replace('/', '-') # + "_" + padding

if __name__ == '__main__':
    
    # hyperparams
    # python 00_extract_embeddings -- dataset sst2 --checkpoint textattack/bert-base-uncased-SST-2
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, help='name of model checkpoint', default='bert-base-uncased')
    parser.add_argument('--ngrams', type=int, help='dimensionality of ngrams', default=1)
    parser.add_argument('--subsample', type=int, help='must be -1! subsampling no longer supported', default=-1)
    parser.add_argument('--dataset', type=str, help='which dataset to fit', default='sst2') # sst2, imdb, emotion, rotten_tomatoes
    args = parser.parse_args()
    args.padding = True # 'max_length' # True
    print('\n\nembed_dset hyperparams', vars(args), '\n\n')
    
    # check if cached
    dir_name = get_dir_name(args)
    save_dir = oj(config.data_dir, args.dataset, dir_name)
    if os.path.exists(save_dir):
        print('aready ran', save_dir)
        exit(0)
    
    # set up model
    nlp = English()
    simple_tokenizer = nlp.tokenizer # for our word-finding
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint) # for actually passing things to the model
    if 'distilbert' in args.checkpoint:
        model = DistilBertModel.from_pretrained(args.checkpoint)
    elif 'bert-base' in args.checkpoint or 'BERT' in args.checkpoint:
        model = BertModel.from_pretrained(args.checkpoint)
    
    
    # set up data
    dataset, args = data.process_data_and_args(args)
        
    # run 
    with torch.no_grad():
        embedded_dataset = dataset.map(embed_and_sum_function) #, batched=True)
    
    # save
    os.makedirs(save_dir, exist_ok=True)
    embedded_dataset.save_to_disk(save_dir)
    """
    for k in ['train', 'validation', 'test']:
        embs = np.array(dataset[k]['embs']).squeeze()
        pkl.dump(embs, open(oj(save_dir, 'embs_' + k + '.pkl'), 'wb'))
    """
