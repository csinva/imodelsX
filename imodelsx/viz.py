import os
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import join
import sklearn
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import sys
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.tree._tree import Tree
from sklearn import __version__

DSETS_RENAME_DICT = {
    'emotion': 'Emotion',
    'sst2': 'SST2',
    'tweet_eval': 'Tweet (Hate)',
    'imdb': 'IMDB',
    'rotten_tomatoes': 'Rotten tomatoes',
    'financial_phrasebank': 'Financial phrasebank',
}

DSETS_ABBR_RENAME_DICT = {
    'emotion': 'Emotion',
    'sst2': 'SST2',
    'tweet_eval': 'Tweet (Hate)',
    'imdb': 'IMDB',
    'rotten_tomatoes': 'RT',
    'financial_phrasebank': 'FPB',
}

MODELS_RENAME_DICT = {
    'decision_tree': 'CART',
    'manual_tree': 'TreePrompt',
    'manual_ensemble': 'Ensemble (best-first)',
    'manual_boosting': 'Ensemble (boosting)',
    'manual_gbdt': 'TreePrompt (GBDT)',
    'single_prompt': 'Single prompt',
}

CHECKPOINTS_RENAME_DICT = {
    'EleutherAI/gpt-j-6B': 'GPT-J (6B)',
    'gpt2': 'GPT-2 (117M)',
    'gpt2-medium': 'GPT-2 (345M)',
    'gpt2-large': 'GPT-2 (762M)',
    'gpt2-xl': 'GPT-2 (1.5B)',
    'gpt-3.5-turbo': 'ChatGPT',
    'gpt-4-0314': 'GPT-4',
    'text-davinci-003': 'GPT-3',
    'llama_7b': 'LLAMA (7B)',
    'meta-llama/Llama-2-7b-hf': 'LLAMA-2 (7B)',
    'meta-llama/Llama-2-13b-hf': 'LLAMA-2 (13B)',
}

METRICS_RENAME_DICT = {
    'accuracy': 'Accuracy',
    'f1': 'F1',
    'precision': 'Precision',
    'recall': 'Recall',
    'roc_auc': 'ROC AUC',
}


def _extract_arrays_from_llm_tree(llm_tree, dtreeviz_dummies):
    """Takes in an LLM tree and recursively converts it to arrays
    that we can later use to build a sklearn decision tree object
    """
    TreeData = namedtuple(
        'TreeData', 'left_child right_child feature threshold impurity n_node_samples weighted_n_node_samples')
    tree_data = TreeData(
        left_child=[],
        right_child=[],
        feature=[],
        threshold=[],
        impurity=[],
        n_node_samples=[],
        weighted_n_node_samples=[],
    )

    value_sklearn_array = []
    strs_array = []
    node_id_counter = 0
    node_queue = [llm_tree.root_]
    while len(node_queue) > 0:
        node = node_queue.pop(0)

        # add a dummy node
        if node is None:
            tree_data.left_child.append(-1)
            tree_data.right_child.append(-1)
            tree_data.feature.append(0)
            tree_data.threshold.append(0)
            tree_data.impurity.append(-1)
            if dtreeviz_dummies:
                tree_data.n_node_samples.append(1)
                tree_data.weighted_n_node_samples.append(1)
                value_sklearn_array.append(np.array([0, 1]).astype(float))
            else:
                tree_data.n_node_samples.append(0)
                tree_data.weighted_n_node_samples.append(0)
                value_sklearn_array.append(np.array([0, 0]).astype(float))
            strs_array.append('')
            continue

        node_id_left = -1
        node_id_right = -1
        feature = -2
        threshold = -2
        value_sklearn = np.array(node.n_samples).astype(float)

        has_children = node.child_left is not None or node.child_right is not None
        if has_children:
            # this should correspond to feature_names later...
            feature = len(tree_data.feature)
            threshold = 0.5  # node.threshold

            # if node.child_left is not None:
            node_id_left = node_id_counter + 1  # node.child_left.node_id
            node_id_counter += 1

            # if node.child_right is not None:
            node_id_right = node_id_counter + 1  # node.child_right.node_id
            node_id_counter += 1

        # print(feature, node)
        tree_data.left_child.append(node_id_left)
        tree_data.right_child.append(node_id_right)
        tree_data.feature.append(feature)
        tree_data.threshold.append(threshold)
        tree_data.impurity.append(node.acc)
        # tree_data.impurity.append(node.impurity)
        tree_data.n_node_samples.append(np.sum(value_sklearn))
        value_sklearn_array.append(value_sklearn)
        tree_data.weighted_n_node_samples.append(
            np.sum(value_sklearn))  # TODO add sample weights
        node_queue.append(node.child_left)
        node_queue.append(node.child_right)

        strs_array.append(node.get_str_simple())

    return tree_data, np.array(value_sklearn_array), strs_array


def extract_sklearn_tree_from_llm_tree(
    llm_tree, n_classes,
    with_leaf_predictions=False,
    dtreeviz_dummies=False,
):
    """Takes in a Tree model and convert tree tree_num to a sklearn decision tree
    """

    tree_data_namedtuple, value_sklearn_array, strs_array = \
        _extract_arrays_from_llm_tree(
            llm_tree, dtreeviz_dummies=dtreeviz_dummies)
    # for k in tree_data_namedtuple._fields:
    # print(k)
    # print(tree_data_namedtuple.__getattribute__(k))

    # manipulate tree_data_namedtuple into the numpy array of tuples
    # that sklearn expects for use with __setstate__()
    df_tree_data = pd.DataFrame(tree_data_namedtuple._asdict())
    tree_data_list_of_tuples = list(
        df_tree_data.itertuples(index=False, name=None))
    _dtypes = np.dtype([('left_child', 'i8'), ('right_child', 'i8'), ('feature', 'i8'), ('threshold',
                       'f8'), ('impurity', 'f8'), ('n_node_samples', 'i8'), ('weighted_n_node_samples', 'f8')])

    tree_data_array = np.array(tree_data_list_of_tuples, dtype=_dtypes)

    # reshape value_sklearn_array to match the expected shape of (n_nodes,1,2) for values
    value_sklearns = value_sklearn_array.reshape(
        value_sklearn_array.shape[0], 1, value_sklearn_array.shape[1])

    if n_classes == 1:
        value_sklearns = np.ascontiguousarray(value_sklearns[:, :, 0:1])

    # get the max_depth
    def get_max_depth(node):
        if node is None:
            return -1
        else:
            return 1 + max(get_max_depth(node.child_left), get_max_depth(node.child_right))

    max_depth = get_max_depth(llm_tree.root_)
    # max_depth = 4

    # get other variables needed for the sklearn.tree._tree.Tree constructor and __setstate__() calls
    # n_samples = np.sum(figs_tree.value_sklearn)
    node_count = len(tree_data_array)
    features = np.array(tree_data_namedtuple.feature)
    n_features = np.unique(features[np.where(0 <= features)]).size
    n_classes_array = np.array([n_classes], dtype=int)
    n_outputs = 1

    # make dict to pass to __setstate__()
    _state = {
        'max_depth': max_depth,
        'node_count': node_count,
        'nodes': tree_data_array,
        'values': value_sklearns,
        # 'n_features_in_': llm_tree.n_features_in_,
        # WARNING this circumvents
        # UserWarning: Trying to unpickle estimator DecisionTreeClassifier
        # from version pre-0.18 when using version
        # https://github.com/scikit-learn/scikit-learn/blob/53acd0fe52cb5d8c6f5a86a1fc1352809240b68d/sklearn/base.py#L279
        '_sklearn_version': __version__,
    }
    # print('state', _state)

    tree = Tree(n_features=n_features,
                n_classes=n_classes_array, n_outputs=n_outputs)
    # https://github.com/scikit-learn/scikit-learn/blob/3850935ea610b5231720fdf865c837aeff79ab1b/sklearn/tree/_tree.pyx#L677
    tree.__setstate__(_state)

    # add the tree_ for the dt __setstate__()
    # note the trailing underscore also trips the sklearn_is_fitted protections
    _state['tree_'] = tree
    _state['classes_'] = np.arange(n_classes)
    _state['n_outputs_'] = n_outputs

    # construct sklearn object and __setstate__()
    # if isinstance(llm_tree, ClassifierMixin):
    dt = DecisionTreeClassifier(max_depth=max_depth)
    # elif isinstance(llm_tree, RegressorMixin):
    # dt = DecisionTreeRegressor(max_depth=max_depth)

    try:
        dt.__setstate__(_state)
    except:
        raise Exception(
            f'Did not successfully run __setstate__() when translating to {type(dt)}, did sklearn update?')

    if not with_leaf_predictions:
        return dt, strs_array
    else:
        leaf_values_dict = {}

        def _read_node(node):
            if node is None:
                return None
            elif node.left is None and node.right is None:
                leaf_values_dict[node.node_id] = node.value[0][0]
            _read_node(node.left)
            _read_node(node.right)
        _read_node(llm_tree)

        return dt, leaf_values_dict
