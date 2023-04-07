import re
from typing import List
import numpy as np
import os.path
from os.path import join
import pickle as pkl
    

def expand_keyword(
    keyphrase_str: str = 'bad',
    llm_prompt_context: str = '', # ' in the context of movie reviews',
    cache_dir: str=None,
    seed: int=0,
    verbose: bool=False,
):
    """Refine a keyphrase by making a call to gpt-3
    """
    
    # check cache
    if cache_dir is not None:
        if llm_prompt_context == '':
            cache_dir = join(cache_dir, 'base')
        else:
            cache_dir = join(cache_dir, ''.join(llm_prompt_context.split()))
        os.makedirs(cache_dir, exist_ok=True)
        keyphrase_str = keyphrase_str.replace('/', ' ')
        cache_file = join(cache_dir, f'_{keyphrase_str}___{seed}.pkl')
        cache_raw_file = join(cache_dir, f'raw_{keyphrase_str}___{seed}.pkl')
        if os.path.exists(cache_file):
            if verbose:
                print('cached!')
            return pkl.load(open(cache_file, 'rb'))

    prompt = f'Generate 100 concise phrases that are very similar to the keyphrase{llm_prompt_context}:\n'
    prompt += f'Keyphrase: "{keyphrase_str}"\n'
    prompt += '1.'

    import openai    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        temperature=0.1,
        top_p=1,
        frequency_penalty=0.25,  # maximum is 2
        presence_penalty=0,
        stop=["101"]
)
    response_text = response['choices'][0]['text']
    ks = convert_response_to_keywords(response_text)
    if cache_dir is not None:
        pkl.dump(response_text, open(cache_raw_file, 'wb'))
        pkl.dump(ks, open(cache_file, 'wb'))
    return ks


def convert_response_to_keywords(response_text: str, remove_duplicates=True) -> List[str]:
    # clean up the keyphrases
    # (split the string s on any numeric character)
    ks = [
        k.replace('.', '').replace('"', '').strip()
        for k in re.split(r'\d', response_text) if k.strip()
    ]

    # lowercase & len > 2
    ks = [k.lower() for k in ks if len(k) > 2]

    if remove_duplicates:
        ks = list(set(ks))
    return ks

if __name__ == '__main__':
    refined_keyphrase = expand_keyword(
        keyphrase_str='koala',
        # task_context_str=' about movie reviews'
        )
    
    print('Refined keyphrase:', refined_keyphrase)
