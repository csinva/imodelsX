import itertools
from slurmpy import Slurm

# slurm params
partition = 'jsteinhardt' # yugroup, jsteinhardt
num_gpus = 1
s = Slurm("embed_dset", {"partition": partition, "time": "4-0", "gres": f"gpu:{num_gpus}"})

    
##########################################
# main setting finetuned archs
##########################################    
# GLOBAL_PARAMS = {
#     'ngrams': [1, 2, 3, 4, 5, 6, 7],    
#     'layer': ['last_hidden_state_mean'], # 'pooler_output'
# }

# PARAMS_LIST = [
#     {
#         'dataset': ['sst2'],    
#         # 'checkpoint': ['textattack/bert-base-uncased-SST-2'],
#         'checkpoint': ['distilbert-base-uncased-finetuned-sst-2-english'],
#     },
#     {
#         'dataset': ['emotion'],
#         # 'checkpoint': ['nateraw/bert-base-uncased-emotion'],
#         'checkpoint': ['aatmasidha/distilbert-base-uncased-finetuned-emotion'], #        
#     },
#     {
#        'dataset': ['rotten_tomatoes'],
#        # 'checkpoint': ['textattack/bert-base-uncased-rotten_tomatoes'],
#         'checkpoint': ['textattack/distilbert-base-uncased-rotten-tomatoes'], #        
#     },
#     {
#         'dataset': ['tweet_eval'],
#         # 'checkpoint': ['philschmid/BERT-tweet-eval-emotion'],
#         'checkpoint': ['philschmid/DistilBERT-tweet-eval-emotion'],
#     },
#     {
#         'dataset': ['financial_phrasebank'],
#         # 'checkpoint': ['ahmedrachid/FinancialBERT-Sentiment-Analysis'],
#         'checkpoint': ['yseop/distilbert-base-financial-relation-extraction'], #  note this match isn't perfect
#     }
# ]

    
##########################################
# main setting common archs
##########################################    
GLOBAL_PARAMS = {
    'ngrams': [1, 2, 3, 4, 5, 6, 7],    
    'layer': ['last_hidden_state_mean'], # 'pooler_output'
    'checkpoint': ['EleutherAI/gpt-neo-2.7B', 'distilbert-base-uncased', 'bert-base-uncased'],    
}

PARAMS_LIST = [
    {'dataset': ['sst2']},
    {'dataset': ['emotion']},
    {'dataset': ['rotten_tomatoes']},
    {'dataset': ['tweet_eval']},
    {'dataset': ['financial_phrasebank']}
]
    
    
##########################################
# noun_chunks
##########################################
# GLOBAL_PARAMS = {
#     'ngrams': [1, 2],    
#     'layer': ['last_hidden_state_mean'],
#     'parsing': ['noun_chunks'],
# }

# PARAMS_LIST = [
# {
#     'dataset': ['sst2'],    
#     'checkpoint': ['textattack/bert-base-uncased-SST-2'],
# },
# {
#     'dataset': ['emotion'],
#     'checkpoint': ['nateraw/bert-base-uncased-emotion'],
# },
# {
#    'dataset': ['rotten_tomatoes'],
#    'checkpoint': ['textattack/bert-base-uncased-rotten_tomatoes'],
# },
# {
#     'dataset': ['tweet_eval'],
#     'checkpoint': ['philschmid/BERT-tweet-eval-emotion'],       
# },
# {
#     'dataset': ['financial_phrasebank'],
#     'checkpoint': ['ahmedrachid/FinancialBERT-Sentiment-Analysis'],
# },
# ]

for PARAMS in PARAMS_LIST:
    ks = list(PARAMS.keys())
    vals = [PARAMS[k] for k in ks]

    ks2 = list(GLOBAL_PARAMS.keys())
    vals += [GLOBAL_PARAMS[k] for k in ks2]
    ks += ks2

    param_combinations = list(itertools.product(*vals)) # list of tuples


    for i in range(len(param_combinations)):
        param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../00_extract_embeddings.py '    
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        s.run(param_str)
        # print(param_str)


# # imdb (too big)
# PARAMS = {
#     'dataset': ['imdb'],
#     'checkpoint': ['textattack/bert-base-uncased-imdb'], # 'bert-base-uncased', 
# }


"""
# set params directly
checkpoint = 'bert-base-uncased'
ngrams = 1
subsample = 100

# iterate
param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../01_embed_dset.py '    
param_str += '--checkpoint ' + checkpoint + ' '
param_str += '--ngrams ' + str(ngrams) + ' '
param_str += '--subsample ' + str(subsample) + ' '


# execute slurm
s.run(param_str)
"""
