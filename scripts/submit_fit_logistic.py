import itertools
from slurmpy import Slurm

# slurm params
partition = 'yugroup'
s = Slurm("fit_logistic", {"partition": partition, "time": "1-0"})


# set param combos
PARAMS = {
    'subsample': [100, 1000, -1],
    'ngrams': [1, 2, 3, 4, 5, 6, 7, 10],    
    'checkpoint': ['textattack/bert-base-uncased-SST-2'] , #['countvectorizer', 'tfidfvectorizer', 'bert-base-uncased'],
    'all': ['all'],
}
# PARAMS = {
#     'subsample': [100, 1000, -1],
#     'ngrams': [2, 3, 4, 5],    
#     'checkpoint': ['bert-base-uncased'],
#     'all': ['all'],
#     'norm': ['norm'],
# }


ks = PARAMS.keys()
vals = [PARAMS[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples


for i in range(len(param_combinations)):
    param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../03_fit_logistic.py '    
    for j, key in enumerate(ks):
        param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
#     print(param_str)