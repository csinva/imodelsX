from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
import embgam.data as data
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
from embgam.embed import embed_and_sum_function
path_to_current_file = os.path.dirname(os.path.abspath(__file__))


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
