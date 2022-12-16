import pickle as pkl
import os
from collections import defaultdict
import random
import json
import torch
import transformers
from os.path import dirname, join

D3_DIR = dirname(os.path.abspath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def sample_sentences(xs, k, group_id):
    random.shuffle(xs)
    return '\n'.join(['Group %s: %s' % (group_id, x) for x in xs[:k]])


def sort_by_score(d):
    return sorted(d, key=lambda k: d[k], reverse=True)


def get_top_percentile(l, p, min_length=10):
    n = max(int(len(l) * p / 100), min_length)
    return l[:n]


class Proposer:

    def __init__(self, model_name, template_path):
        self.proposer_name = model_name
        self.prompt_template = open(template_path).read().strip()

    def preprocess_texts(self, x2score):
        return [self.normalize(x) for x in sort_by_score(x2score)]

    def create_prompt(self, A_block, B_block):
        prompt = self.prompt_template.format(A_block=A_block, B_block=B_block)
        return prompt

    def propose_hypothesis(self, pos2score, neg2score, hyp_count, num_incontext_samples, temperature):
        raise NotImplementedError

    def normalize(self, x):
        raise NotImplementedError


class GPT3Proposer(Proposer):

    def __init__(self, model_name):
        super(GPT3Proposer, self).__init__(model_name, join(
            D3_DIR, 'templates/gpt3_proposer_template.txt'))
        self.discouraged_toks = [4514, 8094, 33, 40798, 392, 273, 14, 11, 981, 4514,
                                 8094, 1448, 33, 347, 1884, 40798, 290, 392, 273, 393, 14, 1220, 837, 11]
        self.tok = transformers.T5TokenizerFast.from_pretrained('gpt2')

        self.hyps = {
            'max_tokens': 50,
            'n': 1,
            'top_p': 1,
            'engine': self.proposer_name,
            'logit_bias': {i: -100 for i in self.discouraged_toks}
        }

    def propose_hypothesis(self, pos2score, neg2score, hyp_count=90,
                           num_incontext_samples=5, temperature=0.7, percentiles=[10, 20, 100]):
        pos_sorted, neg_sorted = self.preprocess_texts(
            pos2score), self.preprocess_texts(neg2score)
        all_hs = []

        for percentile in percentiles:
            # get the top percentile examples
            pos = get_top_percentile(pos_sorted, percentile)
            neg = get_top_percentile(neg_sorted, percentile)
            all_hs.extend(self.propose_w_pos_neg(
                pos, neg, hyp_count // len(percentiles), num_incontext_samples, temperature))
        return all_hs

    def propose_w_pos_neg(self, pos, neg, hyp_count, num_incontext_samples, temperature):
        returned_hyps = []
        import openai
        openai.api_key = None  # place your OpenAI API key here

        for _ in range(hyp_count):
            try_count = 0

            while try_count < 50:
                A_block = sample_sentences(
                    pos, k=num_incontext_samples, group_id='A')
                B_block = sample_sentences(
                    neg, k=num_incontext_samples, group_id='B')
                prompt = self.create_prompt(A_block, B_block)
                try:
                    response = openai.Completion.create(
                        prompt=prompt,
                        stop=["\n", '.'],
                        temperature=temperature,
                        **self.hyps
                    )
                    h = response['choices'][0]['text'].strip()
                    returned_hyps.append(h)
                    break
                except KeyboardInterrupt:
                    exit(0)
                except Exception as e:
                    print(e)

                try_count += 1
        return returned_hyps

    def normalize(self, x):
        return self.tok.decode(self.tok(x)['input_ids'][:192], skip_special_tokens=True)


class T5Proposer(Proposer):

    def __init__(self, model_name, verbose=True):
        super(T5Proposer, self).__init__(model_name, join(
            D3_DIR, 'templates/t5_ai2_proposer_template.txt'))
        if verbose:
            print('loading model')
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            model_name).half().to(device)
        self.model.eval()
        if verbose:
            print('loading finishes')
        self.tok = transformers.T5TokenizerFast.from_pretrained('t5-small')
        self.discouraged_toks = [[298], [563], [71],
                                 [272], [952], [1531], [3, 87], [3, 6]]

    def normalize(self, x):
        return self.tok.decode(self.tok(x)['input_ids'][:128], skip_special_tokens=True)

    def propose_hypothesis(self, pos2score, neg2score, hyp_count=90, num_incontext_samples=5, temperature=0.85):
        pos_sorted, neg_sorted = self.preprocess_texts(
            pos2score), self.preprocess_texts(neg2score)
        all_hs = []

        for ensemble_method in ['prob', 'logit']:
            for percentile in [10, 20, 100]:
                # get the top percentile examples
                pos = get_top_percentile(pos_sorted, percentile)
                neg = get_top_percentile(neg_sorted, percentile)

                for num_prompt_ensemble in [1, 3, 5]:
                    for _ in range(hyp_count // 18):
                        prompts = []

                        for j in range(num_prompt_ensemble):
                            A_block = sample_sentences(
                                pos, k=num_incontext_samples, group_id='A')
                            B_block = sample_sentences(
                                neg, k=num_incontext_samples, group_id='B')
                            prompt = self.create_prompt(A_block, B_block)
                            prompts.append(prompt)

                        hs = self.inference_on_ensemble_prompts(
                            prompts, 1, temperature, ensemble_method)
                        all_hs.extend(hs)
        return all_hs

    def inference_on_ensemble_prompts(self, prompts, n, temperature, ensemble_method):
        """Note: this is currently ignoring ensembling!!!
        """
        input_dict = self.tok(prompts, return_tensors="pt",
                              padding=True).to(device)
        input_dict["bad_words_ids"] = self.discouraged_toks
        generated_tokens = self.model.generate(**input_dict,
                                               output_scores=True,
                                               return_dict_in_generate=True,
                                               do_sample=True,
                                               top_k=0, num_return_sequences=n, temperature=temperature,
        )
                                            #    ensemble_sample=True, ensemble_method=ensemble_method)
        completions = self.tok.batch_decode(
            generated_tokens.sequences, skip_special_tokens=True)
        return completions[:n]


def init_proposer(proposer_name):
    if proposer_name[:2] == 't5':
        return T5Proposer(proposer_name[2:])
    if proposer_name[:4] == 'gpt3':
        return T5Proposer(proposer_name[4:])
    raise Exception('Proposer %s has not been implemented' % proposer_name)
