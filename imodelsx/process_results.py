import argparse
import sys
import os.path
from os.path import dirname, join
from os.path import join
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import sys
repo_dir = dirname(dirname(os.path.abspath(__file__)))

def get_results_df(results_dir, use_cached=False) -> pd.DataFrame:
    """Load results from a directory of experiments, 
    each experiments is a row in the dataframe
    """
    fname = join(results_dir, 'results_aggregated.pkl')
    if use_cached and os.path.exists(fname):
        return pd.read_pickle(fname)
    dir_names = sorted([fname
                        for fname in os.listdir(results_dir)
                        if os.path.isdir(join(results_dir, fname))
                        and os.path.exists(join(results_dir, fname, 'results.pkl'))
                        ])
    results_list = []
    for dir_name in tqdm(dir_names):
        ser = pd.Series(
            pkl.load(open(join(results_dir, dir_name, 'results.pkl'), "rb")))
        results_list.append(ser)
    r = pd.concat(results_list, axis=1).T.infer_objects()
    r.to_pickle(fname)
    return r

def get_main_args_list(experiment_filename='01_train_model.py'):
    """Returns main arguments from the argparser used by an experiments script

    Params
    ------
    fname: str
        Full path + name of the experiments script, e.g. /home/user/tree-prompt/experiments/01_train_model.py
    """
    if experiment_filename.endswith('.py'):
        experiment_filename = experiment_filename[:-3]
    # sys.path.append(join(repo_dir, 'experiments'))
    sys.path.append(os.path.dirname(experiment_filename))
    train_script = __import__(os.path.basename(experiment_filename))
    args = train_script.add_main_args(argparse.ArgumentParser()).parse_args([])
    return list(vars(args).keys())

def fill_missing_args_with_default(df, experiment_filename='01_train_model.py'):
    """Returns main arguments from the argparser used by an experiments script
    Params
    ------

    fname: str
        Full path + name of the experiments script, e.g. /home/user/tree-prompt/experiments/01_train_model.py
    """
    if experiment_filename.endswith('.py'):
        experiment_filename = experiment_filename[:-3]
    sys.path.append(os.path.dirname(experiment_filename))
    train_script = __import__(os.path.basename(experiment_filename))
    parser = train_script.add_main_args(argparse.ArgumentParser())
    parser = train_script.add_computational_args(parser)
    args = parser.parse_args([])
    args_dict = vars(args)
    for k, v in args_dict.items():
        if k not in df.columns:
            df[k] = v
        df[k] = df[k].fillna(v)
    return df