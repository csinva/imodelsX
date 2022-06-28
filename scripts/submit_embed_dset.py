import itertools
from slurmpy import Slurm

# slurm params
partition = 'jsteinhardt' # yugroup, jsteinhardt
num_gpus = 1
s = Slurm("embed_dset", {"partition": partition, "time": "5-0", "gres": f"gpu:{num_gpus}"})

# sst 
# PARAMS = {
#     'subsample': [-1],
#     'ngrams': [1, 2, 3, 4, 5, 6, 7, 10],    
#     'checkpoint': ['textattack/bert-base-uncased-SST-2'], #'bert-base-uncased'
#     'dataset': ['sst2'],
# }

# imdb
PARAMS = {
    'subsample': [-1], #, 1000, 100],
    'ngrams': [1, 2, 3, 4, 5, 6, 7],
    'checkpoint': ['textattack/bert-base-uncased-imdb'], #'textattack/bert-base-uncased-SST-2']
    'dataset': ['imdb'],
}

# emotion
# PARAMS = {
#     'subsample': [-1], #, 1000, 100],
#     'ngrams': [1, 2, 3, 4, 5, 6, 7],    
#     # 'checkpoint': ['textattack/bert-base-uncased-imdb'], #'textattack/bert-base-uncased-SST-2'], #'bert-base-uncased'],
#     'checkpoint': ["bert-base-uncased", 'nateraw/bert-base-uncased-emotion'],
#     'dataset': ['emotion'],
# }

# rotten_tomatoes
# PARAMS = {
#    'subsample': [-1], #, 1000, 100],
#    'ngrams': [1, 2, 3, 4, 5, 6, 7],    
#    'checkpoint': ["bert-base-uncased", 'textattack/bert-base-uncased-rotten_tomatoes'],
#    'dataset': ['rotten_tomatoes'],
#}

ks = PARAMS.keys()
vals = [PARAMS[k] for k in ks]
param_combinations = list(itertools.product(*vals)) # list of tuples


for i in range(len(param_combinations)):
    param_str = '/usr/local/linux/anaconda3.8/bin/python3 ../01_embed_dset.py '    
    for j, key in enumerate(ks):
        param_str += '--' + key + ' ' + str(param_combinations[i][j]) + ' '
    s.run(param_str)
#     print(param_str)





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
