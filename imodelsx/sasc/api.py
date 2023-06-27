from typing import List, Callable, Tuple, Dict
import imodelsx.sasc.m1_ngrams
import imodelsx.sasc.m2_summarize
import imodelsx.sasc.m3_generate
import numpy as np
import pprint
from collections import defaultdict


def explain_module_sasc(
    # get ngram module responses
    text_str_list: List[str],
    mod: Callable[[List[str]], List[float]],
    ngrams: int = 3,
    all_ngrams: bool = True,
    num_top_ngrams: int = 75,
    use_cache: bool = True,
    cache_filename: str = None,
    # generate explanation candidates
    llm_checkpoint: str = "text-davinci-003",
    llm_cache_dir: str = ".llm_cache",
    num_summaries: int = 3,
    num_top_ngrams_to_use: int = 30,
    num_top_ngrams_to_consider: int = 50,
    # generate synthetic strs
    num_synthetic_strs: int = 20,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, List]:
    """

    Parameters
    ----------
    text_str_list: List[str]
        The list of text strings to use to extract ngrams
    mod: Callable[[List[str]], List[float]]
        The module to interpret
    ngrams: int
        The order of ngrams to use (3 is trigrams)
    all_ngrams: bool
        If True, use all ngrams up to ngrams. If False, use only ngrams
    num_top_ngrams: int
        The number of top ngrams to return
    use_cache: bool
        If True, use the cache
    cache_filename: str
        The filename to use for the module ngram cache
    llm_checkpoint: str
        The checkpoint to use for the llm
    llm_cache_dir: str
        The cache directory to use for the llm
    num_summaries: int
        The number of candidate explanations to generate
    num_top_ngrams_to_use: int
        The number of top ngrams to select
    num_top_ngrams_to_consider: int
        The number of top ngrams to consider selecting from
    num_synthetic_strs: int
        The number of synthetic strs to generate
    verbose: bool
        If True, print out progress
    seed: int
        The seed to use for the random number generator

    Returns
    -------
    explanation_dict: Dict[str, List]
        top_explanation_str: str
            The top explanation str
        top_explanation_score: float
            The top explanation score
        explanation_strs: List[str]
            The list of candidate explanation strs (this may have less entries than num_summaries if duplicate explanations are generated)
        explanation_scores: List[float]
            The list of corresponding candidate explanation scores
        ngrams_list: List[str]
            The list of top ngrams
        ngrams_scores: List[float]
            The list of top ngram scores
        strs_relevant: List[List[str]]
            The list of synthetically generated relevant strs
        strs_irrelevant: List[List[str]]
            The list of synthetically generated irrelevant strs
    """

    explanation_dict = defaultdict(list)

    # compute scores for each ngram
    (
        ngrams_list,
        ngrams_scores,
    ) = imodelsx.sasc.m1_ngrams.explain_ngrams(
        text_str_list=text_str_list,
        mod=mod,
        ngrams=ngrams,
        all_ngrams=all_ngrams,
        num_top_ngrams=num_top_ngrams,
        use_cache=use_cache,
        cache_filename=cache_filename,
    )
    explanation_dict["ngrams_list"] = ngrams_list
    explanation_dict["ngrams_scores"] = ngrams_scores

    # compute explanation candidates
    llm = imodelsx.llm.get_llm(llm_checkpoint, llm_cache_dir)
    (
        explanation_strs,
        _,
    ) = imodelsx.sasc.m2_summarize.summarize_ngrams(
        llm,
        ngrams_list,
        num_summaries=num_summaries,
        num_top_ngrams_to_use=num_top_ngrams_to_use,
        num_top_ngrams_to_consider=num_top_ngrams_to_consider,
        seed=seed,
    )
    explanation_dict["explanation_strs"] = explanation_strs

    # score explanation candidates on synthetic data
    for explanation_str in explanation_strs:
        strs_rel, strs_irrel = imodelsx.sasc.m3_generate.generate_synthetic_strs(
            llm,
            explanation_str=explanation_str,
            num_synthetic_strs=num_synthetic_strs,
            verbose=verbose,
        )
        explanation_dict["strs_relevant"].append(strs_rel)
        explanation_dict["strs_irrelevant"].append(strs_irrel)

        # evaluate synthetic data (higher score is better)
        explanation_dict["explanation_scores"].append(
            np.mean(mod(strs_rel)) - np.mean(mod(strs_irrel))
        )

    # sort everything by scores
    sort_inds = np.argsort(explanation_dict["explanation_scores"])[::-1]
    ks = list(explanation_dict.keys())
    for k in [
        "explanation_strs",
        "explanation_scores",
        "strs_relevant",
        "strs_irrelevant",
    ]:
        explanation_dict[k] = [explanation_dict[k][i] for i in sort_inds]
    for k in ["explanation_strs", "explanation_scores"]:
        explanation_dict["top_" + k[:-1]] = explanation_dict[k][0]

    return explanation_dict


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
