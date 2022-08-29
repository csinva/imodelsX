from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
import data
import numpy as np
import pickle as pkl
import os
import os.path
from os.path import join as oj
from spacy.lang.en import English
import spacy
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
    args.parsing: str, ""
    """
    
    seqs = []
    
    # unigrams
    if args.ngrams == 1:
        seqs += [str(x) for x in simple_tokenizer(sentence)]
    
    # all ngrams in loop
    else:
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
    
    # add noun_chunks which at least have a space in them
    if args.parsing == 'noun_chunks':
        doc = nlp_chunks(sentence)
        seqs += [
            chunk.text for chunk in doc.noun_chunks
            if ' ' in chunk.text
        ]
    return seqs


def preprocess_gpt_token_batch(seqs):
    """Preprocess token batch with token strings of different lengths
    Add attention mask here
    """
    batch_size = len(seqs)
    
    token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in seqs]
    prompt_lengths = [len(s) for s in token_ids]
    max_prompt_len = max(prompt_lengths)
    
    # use 0 as padding id, shouldn't matter (snippet from here https://github.com/huggingface/transformers/issues/3021)
    padded_tokens = [tok_ids + [0] * (max_prompt_len - len(tok_ids)) for tok_ids in token_ids]
    input_ids = torch.LongTensor(padded_tokens)
    attn_mask = torch.zeros(input_ids.shape).long()
    for ix, tok_ids in enumerate(token_ids):
        attn_mask[ix][:len(tok_ids)] = 1
    
    # tokens = tokenizer(seqs, truncation=True, return_tensors="pt")
    return {'input_ids': input_ids, 'attention_mask': attn_mask}

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
        
 
    if 'bert' in args.checkpoint.lower(): # has up to two keys, 'last_hidden_state', 'pooler_output'
        tokens = tokenizer(seqs, padding=args.padding, truncation=True, return_tensors="pt")
        try:
            tokens = tokens.cuda()
        except:
            print('no cuda!')
            pass
        output = model(**tokens)
        if args.layer == 'pooler_output':
            embs = output['pooler_output'].cpu().detach().numpy()
        elif args.layer == 'last_hidden_state_mean':
            embs = output['last_hidden_state'].cpu().detach().numpy()
            embs = embs.mean(axis=1)
    elif 'gpt' in args.checkpoint.lower():
        # print('tokens', tokens['input_ids'].shape, tokens['input_ids'])
        tokens = preprocess_gpt_token_batch(seqs)
        try:
            tokens = tokens.cuda()
        except:
            pass
        output = model(**tokens)
        
        h = output['hidden_states'] # tuple of (layer x (batch_size, seq_len, hidden_size))
        embs = h[0].cpu().detach().numpy() # (batch_size, seq_len, hidden_size)
        embs = embs.mean(axis=1) # (batch_size, hidden_size)
    print('embs', embs.shape)
    
    # sum over the embeddings
    embs = embs.sum(axis=0).reshape(1, -1)
    if seq_len == 0:
        embs *= 0
    # print('embs', embs.shape)    
    
    return {'embs': embs, 'seq_len': len(seqs)}



if __name__ == '__main__':
    # hyperparams
    # python 00_extract_embeddings.py --dataset sst2 --checkpoint textattack/bert-base-uncased-SST-2
    # python 00_extract_embeddings.py --dataset sst2 --checkpoint distilbert-base-uncased --layer last_hidden_state_mean
    # python 00_extract_embeddings.py --dataset sst2 --layer last_hidden_state_mean --parsing noun_chunks
    # python 00_extract_embeddings.py --dataset sst2 --layer last_hidden_state_mean --checkpoint "EleutherAI/gpt-neo-2.7B"
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--checkpoint', type=str, help='name of model checkpoint', default='bert-base-uncased')
    parser.add_argument('--ngrams', type=int, help='dimensionality of ngrams', default=1)
    parser.add_argument('--subsample', type=int, help='must be -1! subsampling no longer supported', default=-1)
    parser.add_argument('--dataset', type=str, help='which dataset to fit', default='sst2') # sst2, imdb, emotion, rotten_tomatoes
    parser.add_argument('--layer', type=str, help='which layer of the model to extract', default='pooler_output') # last_hidden_state_mean
    parser.add_argument('--parsing', type=str, help='extra logic for parsing', default='') # noun_chunks
    args = parser.parse_args()
    args.padding = True # 'max_length' # True
    print('\n\nextract_embeddings hyperparams', vars(args), '\n\n')
    
    # checking
    if 'distilbert' in args.checkpoint.lower() and not args.layer.startswith('last_hidden'):
        raise ValueError('distilbert only has last_hidden output!!!')
    if 'gpt' in args.checkpoint.lower() and not args.layer.startswith('last_hidden'):
        raise ValueError('gpt only has hidden_states output!!!')
    
    # check if cached
    dir_name = data.get_dir_name(args)
    save_dir = oj(config.data_dir, args.dataset, dir_name)
    if os.path.exists(save_dir):
        print('aready ran', save_dir, '\n-------------------SUCCESS------------------------\n\n')
        exit(0)
    
    # set up model
    nlp = English()
    simple_tokenizer = nlp.tokenizer # for our word-finding
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint) # tokenizing for the transformer
    if args.parsing == 'noun_chunks': # for finding noun chunks
        nlp_chunks = spacy.load("en_core_web_sm")
    if 'distilbert' in args.checkpoint.lower():
        model = DistilBertModel.from_pretrained(args.checkpoint)
    elif 'bert-base' in args.checkpoint.lower() or 'BERT' in args.checkpoint:
        model = BertModel.from_pretrained(args.checkpoint)
    elif 'gpt' in args.checkpoint.lower():
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint, output_hidden_states=True)
    try:
        model = model.cuda()
    except:
        pass
        
    
    
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
    
    print(save_dir, '\n', '\n-------------------SUCCESS------------------------\n\n')
