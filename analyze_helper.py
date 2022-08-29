from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import numpy as np
import data
import os.path
from datasets import load_from_disk
import pickle as pkl
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import dvu
dvu.set_style()
import pandas as pd
import config
from os.path import join as oj
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)


def average_seeds(rs):
    varying_cols = ['seed', 'acc_train', 'acc_val']
    rg = rs.groupby(by=[x for x in rs.columns
                       if x not in varying_cols])
    rr = deepcopy(rg).mean().reset_index() #.mean() #.mean().reset_index()
    rr_sem = deepcopy(rg).sem().reset_index()

    # emotion didn't run properly for subsample = 100
    # for g in rg.groups:
    #     num_seeds = rg.groups[g].shape[0]
    #     if num_seeds < 3:
    #         print('only ', num_seeds, 'seeds for ', g)

    for col in ['acc_train', 'acc_val']:
        rr[col + '_sem'] = rr_sem[col]
        rr[col + '_print'] = (100 * rr[col]).round(1).astype(str) + '\% $\pm$ ' + (100*rr[col + '_sem']).round(2).astype(str) + '\%'
    return rr, rr_sem

def bold_extreme_values(data):
    format_string="%.2f"
    max_=True
    if max_:
        extrema = data != data.max()
    else:
        extrema = data != data.min()
    bolded = data.apply(lambda x : "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x : format_string % x)
    return formatted.where(extrema, bolded) 