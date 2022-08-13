import itertools
from slurmpy import Slurm

# slurm params
partition = 'high'
s = Slurm("fit_logistic", {"partition": partition, "time": "1-0"})


# set param combos
PARAMS = {
    'subsample': [100, 1000, -1],
    'ngrams': [1, 2, 3, 4, 5, 6, 7],    
    'dataset': [
#         'sst2',
        'emotion',
        # 'tweet_eval',
        # 'rotten_tomatoes',             
        # 'financial_phrasebank',
        # 'imdb', # too big                   # 'textattack/bert-base-uncased-imdb', # too big
    ],    
    'checkpoint': [
        # 'countvectorizer',
        # 'tfidfvectorizer',
        # 'bert-base-uncased',
        'distilbert-base-uncased',

       # distilbert finetuned
       # 'distilbert-base-uncased-finetuned-sst-2-english',
       'aatmasidha/distilbert-base-uncased-finetuned-emotion',      
        # 'philschmid/DistilBERT-tweet-eval-emotion',                   
        # 'textattack/distilbert-base-uncased-rotten-tomatoes',             
        # 'yseop/distilbert-base-financial-relation-extraction',                   
                   
           # bert models                   
#         'textattack/bert-base-uncased-SST-2',
#         'nateraw/bert-base-uncased-emotion',
#            'philschmid/BERT-tweet-eval-emotion'
        # 'textattack/bert-base-uncased-rotten_tomatoes',         
        # 'ahmedrachid/FinancialBERT-Sentiment-Analysis',
    ],
    'all': ['all'],
    'layer': ['last_hidden_state_mean'],
}

ks = PARAMS.keys()
vals = [PARAMS[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples


for i in range(len(param_combinations)):
    param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../03_fit_logistic.py '    
    for j, key in enumerate(ks):
        param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
#     print(param_str)