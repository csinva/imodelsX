from os.path import join as oj
import os
from os.path import dirname
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# main_dir = '/scratch/users/vision/chandan/embedded-ngrams/'
# main_dir = '/home/chansingh/mntv1/embedded-ngrams'
# main_dir = '/scratch/embgrams/embedded-ngrams'
main_dir = oj(repo_dir, 'tmp')
data_dir = oj(main_dir, 'data')
results_dir = oj(main_dir, 'results')
misc_dir = oj(main_dir, 'misc')
for d_str in [data_dir, results_dir, misc_dir]:
    os.makedirs(d_str, exist_ok=True)