import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def load_huggingface_dataset(
    dataset_name: str, subsample_frac: float = 1.0,
    binary_classification: bool = False,
    return_lists: bool = False,
):
    """Load text dataset from huggingface (with train/validation splits) + return the relevant dataset key
    Params
    ------
    binary_classification: bool
        Whether to convert a multiclass task into a binary one
        Unless this function is modified, will take the class number with the lowest to indexes

    Some examples       |    n_train     |    n_classes
    rotten_tomatoes     |    ~9k         |    2
    sst2                |    ~68k        |    2
    tweet_eval          |    ~10k        |    2
    financial_phrasebank|    ~2.3k       |    3
    emotion             |    ~18k        |    6
    """
    # load dset
    if dataset_name == 'tweet_eval':
        dset = datasets.load_dataset('tweet_eval', 'hate')
    elif dataset_name == 'financial_phrasebank':
        train = datasets.load_dataset('financial_phrasebank', 'sentences_75agree',
                                      revision='main', split='train')
        idxs_train, idxs_val = train_test_split(
            np.arange(len(train)), test_size=0.33, random_state=13)
        dset = datasets.DatasetDict()
        dset['train'] = train.select(idxs_train)
        dset['validation'] = train.select(idxs_val)
    else:
        dset = datasets.load_dataset(dataset_name)

    # process dset
    dataset_key_text = 'text'
    if dataset_name == 'sst2':
        dataset_key_text = 'sentence'
    elif dataset_name == 'financial_phrasebank':
        dataset_key_text = 'sentence'
    elif dataset_name == 'imdb':
        del dset['unsupervised']
        dset['validation'] = dset['test']

    # subsample data
    if subsample_frac > 0:
        n = len(dset['train'])
        dset['train'] = dset['train'].select(np.random.choice(
            range(n), replace=False,
            size=int(n * subsample_frac)
        ))

    # convert to binary classifications
    if binary_classification and len(np.unique(dset['train']['label'])) > 2:
        if dataset_name == 'financial_phrasebank':
            labels_to_keep_remap = {
                0: 1, # negative
                2: 0, # positive
            }
        elif dataset_name == 'emotion':
            labels_to_keep_remap = {
                0: 0, # sadness
                1: 1, # joy
            }
        else:
            labels_to_keep_keys = np.sort(np.unique(dset['train']['label']))[:2]
            labels_to_keep_remap = {
                labels_to_keep_keys[i]: i for i in range(2)
            }
        
        # filter dset labels to only keep these labels
        dset['train'] = dset['train'].filter(lambda ex: ex["label"] in labels_to_keep_remap)
        dset['validation'] = dset['validation'].filter(lambda ex: ex["label"] in labels_to_keep_remap)

        # map these labels to 0/1
        dset['train'] = dset['train'].map(lambda ex: {'label': labels_to_keep_remap[ex['label']]})
        dset['validation'] = dset['validation'].map(lambda ex: {'label': labels_to_keep_remap[ex['label']]})
    
    if return_lists:
        X_train_text= dset['train'][dataset_key_text]
        y_train = np.array(dset['train']['label'])
        X_test_text = dset['validation'][dataset_key_text]
        y_test = np.array(dset['validation']['label'])
        return X_train_text, X_test_text, y_train, y_test
    else:
        return dset, dataset_key_text


def convert_text_data_to_counts_array(dset, dataset_key_text):
    v = CountVectorizer()
    X_train = v.fit_transform(dset['train'][dataset_key_text])
    y_train = dset['train']['label']
    X_test = v.transform(dset['validation'][dataset_key_text])
    y_test = dset['validation']['label']
    feature_names = v.get_feature_names_out().tolist()
    return X_train, X_test, y_train, y_test, feature_names

if __name__ == '__main__':
    dset, k = load_huggingface_dataset('emotion', 1, binary_classification=False)
    print(dset)
    print(dset['train'])
    print(np.unique(dset['train']['label']))

    dset, k = load_huggingface_dataset('emotion', 1, binary_classification=True)
    print(dset)
    print(np.unique(dset['train']['label']))