import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English
import imodelsx.data
import imodelsx.augtree.utils

def convert_text_data_to_counts_array(
    X_train, X_test, ngrams=2, all_ngrams=True,
    tokenizer=None,
    ):
    if tokenizer == None:
        tokenizer = imodelsx.augtree.utils.get_spacy_tokenizer()
        
    if all_ngrams:
        ngram_range=(1, ngrams)
    else:
        ngram_range=(ngrams, ngrams)

    v = CountVectorizer(
        ngram_range=ngram_range,
        tokenizer=tokenizer,
        lowercase=True,
        token_pattern=None,
    )
    X_train = v.fit_transform(X_train)
    X_test = v.transform(X_test)
    feature_names = v.get_feature_names_out().tolist()
    return X_train, X_test, feature_names


def get_all_data(args):
    X_train, X_test, y_train, y_test = imodelsx.data.load_huggingface_dataset(
        dataset_name=args.dataset_name, subsample_frac=args.subsample_frac, return_lists=True)
    X_train, X_test, feature_names = \
        convert_text_data_to_counts_array(X_train, X_test)
    X_train, X_cv, y_train, y_cv = train_test_split(
        X_train, y_train, test_size=0.33, random_state=args.seed)
    return X_train, X_cv, X_test, y_train, y_cv, y_test, feature_names
