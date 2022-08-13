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
        # 'emotion',
        'tweet_eval',
        # 'rotten_tomatoes',             
        # 'financial_phrasebank',
        # 'sst2',        
    ],    
    'checkpoint': [
        # 'countvectorizer',
        # 'tfidfvectorizer',
        # 'bert-base-uncased',
        # 'distilbert-base-uncased',

       # distilbert finetuned
       # 'aatmasidha/distilbert-base-uncased-finetuned-emotion',      
        'philschmid/DistilBERT-tweet-eval-emotion',                   
        # 'textattack/distilbert-base-uncased-rotten-tomatoes',           
        # 'yseop/distilbert-base-financial-relation-extraction',           
       # 'distilbert-base-uncased-finetuned-sst-2-english',
        
                   
           # bert models                   
        # 'nateraw/bert-base-uncased-emotion',
           # 'philschmid/BERT-tweet-eval-emotion'
        # 'textattack/bert-base-uncased-rotten_tomatoes',         
        # 'ahmedrachid/FinancialBERT-Sentiment-Analysis',
        # 'textattack/bert-base-uncased-SST-2',
        
    ],
    'all': ['all'],
    'layer': ['last_hidden_state_mean'],
}

        # 'imdb', # too big                   # 'textattack/bert-base-uncased-imdb', # too big


ks = PARAMS.keys()
vals = [PARAMS[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples


for i in range(len(param_combinations)):
    param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../02_fit_logistic.py '    
    for j, key in enumerate(ks):
        param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
#     print(param_str)