from sklearn.tree import DecisionTreeClassifier
import numpy as np
from spacy.lang.en import English
from sklearn.metrics import mean_squared_error
ROOT = 0
LEFT = 1
RIGHT = 2
NEG = 0
POS = 1

def clean_str(s):
    return s.lower().replace('/', '___').strip()

def impurity_mse(y):
    """Lower impurity (closer to 0) is better
    """
    y_mean = np.mean(y)
    return np.mean((y - y_mean)**2)

def impurity_gini(y):
    """Lower impurity (closer to 0) is better
    """
    return 1 - gini_binary(np.mean(y))

def impurity_entropy(y):
    """Lower impurity (closer to 0) is better
    """
    return 1 - entropy_binary(np.mean(y))

def mse_score(y_true, y_pred):
    """Lower is better
    """
    return -mean_squared_error(y_true, y_pred)

def gini_score(y_true, y_pred):
    """Purer (more accurate) is better
    """
    y_pred = y_pred.astype(bool)
    y_pred_sum = y_pred.sum()
    if y_pred_sum == 0 or y_pred_sum == y_pred.size:
        y_mean = y_true.mean()
    else:
        y_mean = y_true[y_pred].mean()
    return gini_binary(y_mean)

def entropy_score(y_true, y_pred):
    """Purer (more accurate) is better
    """
    y_pred = y_pred.astype(bool)
    y_pred_sum = y_pred.sum()
    if y_pred_sum == 0 or y_pred_sum == y_pred.size:
        y_mean = y_true.mean()
    else:
        y_mean = y_true[y_pred].mean()
    return entropy_binary(y_mean)

def gini_binary(y_mean: float) -> float:
    """Higher is better
    {0, 1} -> 1
    {0.5} -> 0.5
    """
    return y_mean**2 + (1 - y_mean)**2

def entropy_binary(y_mean: float) -> float:
    """Lower is better
    """
    return -y_mean * np.log2(y_mean) - (1 - y_mean) * np.log2(1 - y_mean)

def get_gini_impurity_reduction_from_sklearn_stump(m: DecisionTreeClassifier):
    """Calculate gini impurity reduction in first split of model m
    """
    gini_orig = m.tree_.impurity[ROOT]
    gini_left = m.tree_.impurity[LEFT]
    gini_right = m.tree_.impurity[RIGHT]
    frac_samples_left = m.tree_.n_node_samples[LEFT] / m.tree_.n_node_samples[ROOT]
    frac_samples_right = m.tree_.n_node_samples[RIGHT] / m.tree_.n_node_samples[ROOT]
    gini_reduction = gini_orig - \
        (frac_samples_left * gini_left + frac_samples_right * gini_right)
    return gini_reduction


def check_if_feature_contributes_positively_from_sklearn_stump(m: DecisionTreeClassifier):
    """Check if having the feature being positive makes the value increase or decrease
    """
    # look at value for first split of model
    return m.tree_.value[RIGHT][0, NEG] > m.tree_.value[LEFT][0, POS]

def get_spacy_tokenizer(convert_lower=True, use_stemming=False):
    return LLMTreeTokenizer(convert_lower, use_stemming)

class LLMTreeTokenizer:
    def __init__(self, convert_lower, use_stemming):
        self.tok = English()
        self.convert_lower = convert_lower
        self.use_stemming = use_stemming
        if self.use_stemming:
            from nltk.stem.porter import PorterStemmer
            self.stemmer = PorterStemmer()

    def __call__(self, s):
        if self.convert_lower:
            s = s.lower()
        strs = [str(x) for x in self.tok(s)]
        if self.use_stemming:
            strs = [self.stemmer.stem(x) for x in strs]
        return strs
