from typing import Any, List, Mapping, Optional, Tuple, Callable
import numpy as np
from imodelsx.llm import get_llm


def summarize_ngrams(
    llm: Callable[[str], str],
    ngrams_list: List[str],
    num_summaries: int = 2,
    prefix_str="Here is a list of phrases:",
    suffix_str="What is a common theme among these phrases?\nThe common theme among these phrases is",
    num_top_ngrams_to_use: int = 30,
    num_top_ngrams_to_consider: int = 50,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Refine a keyphrase by making a call to the llm

    Params
    ------
    llm: Callable[[str], str]
        The llm to use
    ngrams_list: List[str]
        The list of ngrams to summarize
    num_summaries: int
        The number of summaries to generate
    prefix_str: str
        The prefix of the prompt string to use for the llm summarization
    suffix_str: str
        The suffix of the prompt string to use for the llm summarization
    num_top_ngrams_to_use: int
        The number of top ngrams to select
    num_top_ngrams_to_consider: int
        The number of top ngrams to consider selecting from
    seed: int
        The seed to use for the random number generator

    Returns
    -------
    summaries: List[str]
        The list of summaries
    summary_rationales: List[str]
        The list of summary rationales (when available)
    """
    rng = np.random.default_rng(seed)

    summaries = []
    summary_rationales = []
    for i in range(num_summaries):
        # randomly sample num_top_ngrams (preserving ordering)
        n_to_consider = min(num_top_ngrams_to_consider, len(ngrams_list))
        n_to_use = min(num_top_ngrams_to_use, n_to_consider)
        idxs = np.sort(rng.choice(n_to_consider, size=n_to_use, replace=False))
        bullet_list_ngrams = "- " + "\n- ".join(np.array(ngrams_list)[idxs])
        prompt = prefix_str + "\n\n" + bullet_list_ngrams + "\n\n" + suffix_str
        if i == 0:
            print("First prompt")
            print(prompt)
        summary = llm(prompt)

        # clean up summary
        summary, summary_rationale = clean_summary(summary)
        summaries.append(summary)
        summary_rationales.append(summary_rationale)

    # remove replicates
    idxs_replicate = [False]
    summaries_running = {summaries[0]}
    for i in range(1, len(summaries)):
        if summaries[i] in summaries_running:
            idxs_replicate.append(True)
        else:
            idxs_replicate.append(False)
        summaries_running.add(summaries[i])
    summaries = [s for i, s in enumerate(summaries) if not idxs_replicate[i]]
    summary_rationales = [
        s for i, s in enumerate(summary_rationales) if not idxs_replicate[i]
    ]

    return summaries, summary_rationales


def clean_summary(summary: str):
    summary = summary.strip().lower()

    # keep removing unnecessary prefixes
    modified_str = True
    while modified_str:
        modified_str = False
        for k in [
            "that",
            "they",
            "are",
            "all",
            "contain",
            "the",
            "use of",
            "related to",
            "involve",
            "some form of",
            "some kind of",
            "the use of",
            "describe",
            "refer to",
            "words",
            "word",
            "used to",
            "relate to",
        ]:
            if summary.startswith(k):
                summary = summary[len(k) :].strip()
                modified_str = True

    # remove unnecessary suffix
    if summary.endswith("."):
        summary = summary[:-1]

    # remove quotation marks
    summary = summary.replace('"', "")

    # sometimes summary comes with a rationale, e.g.
    # Summary: [involve people and places.]
    # Rationale: [Many of the phrases involve someone saying something, or someone being somewhere. There are also references to family members, such as "father's getting" and "grandfather caught a"]
    if ". " in summary:
        summary_clean = summary[: summary.index(". ")].strip()
        summary_rationale = summary[summary.index(". ") + 1 :].strip()
    else:
        summary_clean = summary
        summary_rationale = ""

    return summary_clean, summary_rationale


if __name__ == "__main__":
    summary_clean, summary_rationale = clean_summary(
        "relate to some form of science or research. Many of the phrases refer to specific scientific fields such as actuarial science, transracial adoption, and cognitive ability. Other phrases refer to the effects of certain phenomena, such as prozac and epilepsy"
    )
    # print('Clean:', repr(summary_clean))
    # print('Rationale:', repr(summary_rationale))
    llm = get_llm(checkpoint="text-davinci-003")

    summaries, summary_rationales = summarize_ngrams(
        llm, ["cat", "dog", "bird", "elephant", "cheetah"]
    )
    print("Summaries:", summaries)
    print("Rationales:", summary_rationales)
