import matplotlib.pyplot as plt
from os.path import join as oj
import seaborn as sns
import string
import config
import pandas as pd
from transformers import BertModel, DistilBertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import datasets
import numpy as np
import embgam.data as data
import os.path
from datasets import load_from_disk
import pickle as pkl
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
import dvu

dvu.set_style()

pd.set_option("display.max_rows", None)


def average_seeds(rs):
    varying_cols = ["seed", "acc_train", "acc_val"]
    rg = rs.groupby(by=[x for x in rs.columns if x not in varying_cols])
    rr = deepcopy(rg).mean().reset_index()  # .mean() #.mean().reset_index()
    rr_sem = deepcopy(rg).sem().reset_index()

    # emotion didn't run properly for subsample = 100
    # for g in rg.groups:
    #     num_seeds = rg.groups[g].shape[0]
    #     if num_seeds < 3:
    #         print('only ', num_seeds, 'seeds for ', g)

    for col in ["acc_train", "acc_val"]:
        rr[col + "_sem"] = rr_sem[col]
        rr[col + "_print"] = (
            (100 * rr[col]).round(1).astype(str)
            + "\% $\pm$ "
            + (100 * rr[col + "_sem"]).round(2).astype(str)
            + "\%"
        )
    return rr, rr_sem


def bold_extreme_values(data):
    format_string = "%.2f"
    max_ = True
    if max_:
        extrema = data != data.max()
    else:
        extrema = data != data.min()
    bolded = data.apply(lambda x: "\\textbf{%s}" % format_string % x)
    formatted = data.apply(lambda x: format_string % x)
    return formatted.where(extrema, bolded)


def corrplot_max_abs_unigrams(df, embs):
    def get_idxs_largest_abs_coefs(unigrams, tot_counts, coef, percentile=99.5):
        idxs_punc = np.array(
            list(
                map(
                    lambda s: all(
                        c.isdigit() or c in string.punctuation for c in s),
                    unigrams,
                )
            )
        )
        idxs_count_large = tot_counts > np.percentile(tot_counts, percentile)

        cs = np.abs(coef).flatten()
        idxs_pred = cs >= np.percentile(cs, percentile)

        idxs = (idxs_pred | idxs_count_large) & ~idxs_punc
        return idxs

    idxs = get_idxs_largest_abs_coefs(
        df["unigram"], df["tot_counts"].values, df["coef"].values, percentile=99.5
    )
    es = pd.DataFrame(embs[idxs].T, columns=df["unigram"].values[idxs])
    sims = es.corr()

    # def coef_colors(coef):
    #     if coef >= 0:
    #         return 'green'
    #     else:
    #         return 'purple'

    plt.figure(figsize=(12, 12))
    vabs = np.max(np.abs(sims))
    cm = sns.diverging_palette(10, 240, as_cmap=True)
    cg = sns.clustermap(
        sims,
        cmap=cm,
        center=0.0,
        dendrogram_ratio=0.01,
        cbar_pos=(0.7, 0.7, 0.05, 0.15),
        cbar_kws={"label": "Correlation between embeddings"},
        #                     row_colors=list(map(coef_colors, coef[idxs])),
        #                     row_colors=list(map(cm, m.coef_.flatten()[idxs])),
        #                     row_colors=list(map(cm, np.log(tot_counts[idxs]) / max(np.log(tot_counts[idxs])))),
        #                     yticklabels=3 # how often to plot yticklabels
    )

    cg.ax_row_dendrogram.set_visible(False)
    cg.ax_col_dendrogram.set_visible(False)

    # mask
    mask = np.triu(np.ones_like(sims))
    values = cg.ax_heatmap.collections[0].get_array().reshape(sims.shape)
    new_values = np.ma.array(values, mask=mask)
    cg.ax_heatmap.collections[0].set_array(new_values)
    cg.ax_heatmap.yaxis.set_ticks_position("left")

    xaxis = cg.ax_heatmap.get_xaxis()
    xticklabels = xaxis.get_majorticklabels()
    # plt.tight_layout()
    cg.savefig("results/unigrams_sim.pdf")


def get_bert_coefs(embs, cached_model=oj(config.repo_dir, 'results/sst_bert_finetuned_ngrams=2.pkl')):
    ex_model = pkl.load(open(cached_model, "rb"))  # pickled with python 3.8
    logistic = ex_model.model
    coef_bert = logistic.coef_.squeeze()
    return embs @ coef_bert


def add_bert_coefs(
    d, df, embs, embs2, cached_model=oj(
        config.repo_dir, 'results/sst_bert_finetuned_ngrams=2.pkl'),
):
    """
    r = data.load_fitted_results(fname_filters=['bert-base', 'sub=-1'],
                                dset_filters=['sst2'],
                                drop_model=False)
    ex_model = r[(r.checkpoint == 'textattack/bert-base-uncased-SST-2') & (r.ngrams == 2)].iloc[0]
    """

    df["bert_coef_unigram"] = get_bert_coefs(embs, cached_model)
    d["bert_coef_bigram"] = get_bert_coefs(embs2, cached_model)

    def find_unigram_scores(unigram):
        return df.loc[df["unigram"] == unigram, "bert_coef_unigram"].iloc[0]

    d["bert_coef_unigram1"] = d["unigram1"].apply(find_unigram_scores)
    d["bert_coef_unigram2"] = d["unigram2"].apply(find_unigram_scores)
    return d


def get_sst_dataset():
    class A:
        checkpoint = 'textattack/bert-base-uncased-SST-2'
        dataset = 'sst2'
        padding = True
    args = A()
    dataset, args = data.process_data_and_args(args)
    return dataset
