import re
from typing import Any, List, Mapping, Optional, Tuple, Callable
from imodelsx.llm import get_llm


def generate_synthetic_strs(
    llm: Callable[[str], str],
    explanation_str: str,
    num_synthetic_strs: int = 20,
    template_num: int = 0,
    verbose=True,
) -> Tuple[List[str], List[str]]:
    """Generate text_added and text_removed via call to an LLM.

    Params
    ------
    llm: Callable[[str], str]
        The llm to use
        flan-t5-xxl/opt-iml-max-30b can only generate one sentence before stopping
        EleutherAI/gpt-neox-20b can generate multiple sentences, but they are not faithful to the concept
    explanation_str: str
        The explanation string to use
    num_synthetic_strs: int
        The number of synthetic strings to generate
    template_num: int
        The prompt template number to use

    Returns
    -------
    strs_added: List[str]
        The list of synthetic strings with the explanation scores added
    strs_removed: List[str]
        The list of synthetic strings with the explanation scores removed
    """

    templates = [
        """
Generate {num_synthetic_strs} sentences that {blank_or_do_not}contain the concept of "{concept}":

1. The""",
        """
Generate {num_synthetic_strs} phrases that are {blank_or_do_not}similar to the concept of "{concept}":

1.""",
    ]
    blank_or_do_not_templates = [
        ["", "do not "],
        ["", "not "],
    ]
    template = templates[template_num]
    strs_added = []
    strs_removed = []
    for blank_or_do_not in blank_or_do_not_templates[template_num]:
        prompt = template.format(
            num_synthetic_strs=num_synthetic_strs,
            blank_or_do_not=blank_or_do_not,
            concept=explanation_str,
        )

        # note: this works works with openai model
        # but tends to stop after generating just one text with non-openai
        synthetic_text_numbered_str = llm(prompt, max_new_tokens=400, do_sample=True)
        if verbose:
            print("\n\n---------------\n")
            print(prompt)
            print("\n\n---------------\n")
            print(synthetic_text_numbered_str)
            print("\n\n---------------\n")

        # split the string s on any number followed by period like 1. or 2.
        synthetic_strs_split = re.split(r"\d.", synthetic_text_numbered_str)
        synthetic_strs_split = [s.strip() for s in synthetic_strs_split if s.strip()]
        synthetic_strs = []
        for i in range(len(synthetic_strs_split)):
            s = synthetic_strs_split[i]
            if s.startswith("."):
                s = s[1:]
            synthetic_strs.append(s.strip())
        synthetic_strs = [s for s in synthetic_strs if len(s) > 2]
        if verbose:
            print("synthetic_strs=", synthetic_strs)

        # ks = list(set(ks))  # remove duplicates
        # ks = [k.lower() for k in ks if len(k) > 2] # lowercase & len > 2
        # return ks
        # synthetic_str = synthetic_str.strip()
        # ....

        for s in synthetic_strs:
            if blank_or_do_not == "":
                strs_added.append(s)
            else:
                strs_removed.append(s)
    return strs_added, strs_removed


if __name__ == "__main__":
    # llm = get_llm(checkpoint='EleutherAI/gpt-neox-20b')
    llm = get_llm("text-davinci-003")
    strs_added, strs_removed = generate_synthetic_strs(
        llm,
        explanation_str="anger",
        num_synthetic_strs=20,
        template_num=1,
    )
    print(f"{strs_added=} {strs_removed=}")
