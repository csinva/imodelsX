from os.path import join as oj
import os
from os.path import dirname
repo_dir = dirname(dirname(os.path.abspath(__file__)))



# main_dir = '/scratch/users/vision/chandan/embedded-ngrams/'
main_dir = '/home/chansingh/mntv1/embedded-ngrams'
data_dir = oj(main_dir, 'data')
results_dir = oj(main_dir, 'results')
misc_dir = oj(main_dir, 'misc')