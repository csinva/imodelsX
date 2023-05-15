from imodelsx import explain_module_sasc
import numpy as np
import pprint

if __name__ == "__main__":
    # an overly simple example of a module that responds to the length of a string
    mod = lambda str_list: np.array([len(s) for s in str_list])
    # in this dataset the longest strings happen to be animals, so we are searching for the explanation "animals"
    text_str_list = [
        "red",
        "blue",
        "x",
        "1",
        "2",
        "hippopotamus",
        "elephant",
        "rhinoceros",
    ]
    explanation_dict = explain_module_sasc(
        text_str_list,
        mod,
        ngrams=1,
        num_summaries=2,
        num_top_ngrams=3,
        num_top_ngrams_to_consider=3,
        num_synthetic_strs=2,
    )
    pprint.pprint(explanation_dict)
