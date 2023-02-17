import os
from tqdm import tqdm
import json
import logging
from os.path import join
from dict_hash import sha256

"""Handles utilities for saving/caching.
This file probably does not need to be edited.
"""

def save_json(args={}, save_dir='results', fname='params.json', r={}):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, fname), 'w') as f:
        if isinstance(args, dict):
            json.dump({**args, **r}, f, indent=4)
        else:
            json.dump({**vars(args), **r}, f, indent=4)


def get_save_dir_unique(parser, parser_without_computational_args, args, save_dir_base):
    # ignore computational args
    args_ignore_for_caching = {
        k for k in vars(args)
        if not k in vars(parser_without_computational_args.parse_args([])).keys()
    }

    # ignore non_default args
    args_ignore_default = vars(parser.parse_args([]))

    # get unique hash
    args_vars = vars(args)
    args_cache = {
        k: args_vars[k] for k in args_vars
        if not k in args_ignore_for_caching and
        not args_vars[k] == args_ignore_default[k]
    }
    save_dir_unique_hash = sha256(args_cache)
    save_dir = os.path.join(
        save_dir_base, save_dir_unique_hash) # + save_dir_random_suffix)

    already_cached = check_cached(save_dir_unique_hash, save_dir_base)
    return already_cached, save_dir


def check_cached(save_dir_unique_hash, save_dir, fname_results='results.pkl') -> bool:
    """Check if this configuration has already been run.
    Breaks if parser changes (e.g. changing default values of cmd-line args)
    """
    if not os.path.exists(save_dir):
        return False
    exp_dirs = [d for d in os.listdir(save_dir)
                if os.path.isdir(join(save_dir, d))]
    
    logging.debug('checking for cached run...')
    for exp_dir in tqdm(exp_dirs):
        try:
            if exp_dir.startswith(save_dir_unique_hash):
                params_file = join(save_dir, exp_dir, 'params.json')
                results_final_file = join(save_dir, exp_dir, 'results.pkl')
                if os.path.exists(params_file) and os.path.exists(results_final_file):
                    return True
        except:
            pass
    return False