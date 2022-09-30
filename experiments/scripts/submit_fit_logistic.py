import itertools
from slurmpy import Slurm

# slurm params
partition = 'high'
s = Slurm("fit_logistic", {"partition": partition, "time": "1-0"})

# python ../02_fit_logistic.py --dataset financial_phrasebank --checkpoint distilbert-base-uncased --subsample 1000 --ngrams 1 --all all --layer last_hidden_state_mean --seed 1

# # python ../02_fit_logistic.py --dataset financial_phrasebank --checkpoint distilbert-base-uncased --subsample 1000 --ngrams 1 --all all --layer last_hidden_state_mean --seed 1 --ignore_cache

# python ../02_fit_logistic.py --dataset financial_phrasebank --checkpoint bert-base-uncased --ngrams 1 --all all --layer last_hidden_state_mean --seed 1 --ignore_cache


GLOBAL_PARAMS = {
    'subsample': [-1, 100, 1000], # 100, 1000
    'ngrams': [1, 2, 3, 4, 5, 6, 7],    
    'all': ['all'],    
    'layer': ['last_hidden_state_mean', 'pooler_output'],    
    'seed': [1, 2, 3],    
}

PARAMS_LIST = [
{
    'dataset': ['emotion'],
    'checkpoint': ['nateraw/bert-base-uncased-emotion',
                   'aatmasidha/distilbert-base-uncased-finetuned-emotion'],
},    
{
    'dataset': ['tweet_eval'],
    'checkpoint': ['philschmid/BERT-tweet-eval-emotion',
                   'philschmid/DistilBERT-tweet-eval-emotion'],
},    
{
    'dataset': ['rotten_tomatoes'],
    'checkpoint': ['textattack/bert-base-uncased-rotten_tomatoes',
                   'textattack/distilbert-base-uncased-rotten-tomatoes'],
},     
{
    'dataset': ['financial_phrasebank'],
    'checkpoint': ['ahmedrachid/FinancialBERT-Sentiment-Analysis',
                  'yseop/distilbert-base-financial-relation-extraction'],
},     
{
    'dataset': ['sst2'],
    'checkpoint': ['textattack/bert-base-uncased-SST-2',           
                   'distilbert-base-uncased-finetuned-sst-2-english',],
},     
]

CHECKPOINTS_SHARED = [
    'countvectorizer',
    'tfidfvectorizer',
    'bert-base-uncased',
    'distilbert-base-uncased',
]

for i in range(len(PARAMS_LIST)):
    d = PARAMS_LIST[i]
    d['checkpoint'] = d['checkpoint'] + CHECKPOINTS_SHARED

# print(PARAMS_LIST)


"""
# noun_chunks
# python ../02_fit_logistic.py --dataset sst2 --checkpoint textattack/bert-base-uncased-SST-2 --subsample -1 --ngrams 1 --layer last_hidden_state_mean --seed 1 --parsing noun_chunks

GLOBAL_PARAMS = {
    'subsample': [-1, 100, 1000], # 100, 1000
    'ngrams': [1, 2],        
    'layer': ['last_hidden_state_mean'],    
    'seed': [1, 2, 3],    
    'parsing': ['noun_chunks']
}

PARAMS_LIST = [
{
    'dataset': ['emotion'],
    'checkpoint': ['nateraw/bert-base-uncased-emotion'],
},    
{
    'dataset': ['tweet_eval'],
    'checkpoint': ['philschmid/BERT-tweet-eval-emotion'],
},    
{
    'dataset': ['rotten_tomatoes'],
    'checkpoint': ['textattack/bert-base-uncased-rotten_tomatoes'],
},     
{
    'dataset': ['financial_phrasebank'],
    'checkpoint': ['ahmedrachid/FinancialBERT-Sentiment-Analysis'],
},     
{
    'dataset': ['sst2'],
    'checkpoint': ['textattack/bert-base-uncased-SST-2'],
},     
]
"""


for PARAMS in PARAMS_LIST:
    ks = list(PARAMS.keys())
    vals = [PARAMS[k] for k in ks]

    ks2 = list(GLOBAL_PARAMS.keys())
    vals += [GLOBAL_PARAMS[k] for k in ks2]
    ks += ks2

    param_combinations_all = list(itertools.product(*vals)) # list of tuples
    
    
    # block impossible pairings
    checkpoint_key = ks.index('checkpoint')
    layer_key = ks.index('layer')
    param_combinations = []
    for i in range(len(param_combinations_all)):
        param_tuple = param_combinations_all[i]
        if param_tuple[layer_key] == 'pooler_output':
            if 'bert' in param_tuple[checkpoint_key].lower():
                if 'distilbert' in param_tuple[checkpoint_key].lower():
                    pass
                else:
                    param_combinations.append(param_tuple)
        else:
            param_combinations.append(param_tuple)
    
    print('filtered from', len(param_combinations_all), 'to', len(param_combinations))
    print(checkpoint_key, layer_key)
    for i in range(len(param_combinations)):
        param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../02_fit_logistic.py '    
        for j, key in enumerate(ks):
            param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
        s.run(param_str)
        print(param_str)


# 'imdb', # too big                   # 'textattack/bert-base-uncased-imdb', # too big