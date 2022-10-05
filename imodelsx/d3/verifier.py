import pickle as pkl
import random
import os
import numpy as np
import re
import torch
import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_count = torch.cuda.device_count()
BSIZE = 4
if device_count == 4:
    BSIZE = 8


def normalize(t):
    return re.sub("'(.+)'", r'\1', t.lower())


def qc2input(d):
    return normalize(d['q'] + '\\n' + d['c'])


class T5ZeroShotClfQA(torch.nn.Module):

    def __init__(self, qa_model_name, max_seq_length = 128, half_precision=False):
        super(T5ZeroShotClfQA, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('t5-small')
        self.model = T5ForConditionalGeneration.from_pretrained(qa_model_name)
        if half_precision:
            print('Using half precision')
            self.half_precision = half_precision
            self.model = self.model.half()
        if device == 'cuda':
            self.model.to(device)
        self.vocab = self.tokenizer.get_vocab()
        self.yes_id, self.no_id = self.vocab['▁yes'], self.vocab['▁no']
        self.max_seq_length = max_seq_length
        self.lsm = torch.nn.LogSoftmax(dim=-1)

    def create_batch(self, q_dicts):
        input_strings = [qc2input(d) for d in q_dicts]
        input_strings = [normalize(i) for i in input_strings]
        input_dict = self.tokenizer(input_strings, padding=True, return_tensors="pt",
                                    truncation=True, max_length=self.max_seq_length).to(device)
        return input_dict

    def forward(self, input_dict):
        starts = torch.tensor([[self.model.config.decoder_start_token_id]] * len(input_dict['input_ids'])).to(device)
        output = self.model(**input_dict, decoder_input_ids=starts)
        logits = self.lsm(output.logits[:, 0, [self.no_id, self.yes_id]])
        return logits

    def get_logits_from_input_dict_(self, input_strings):
        input_dict = self.create_batch(input_strings)
        return self.forward(input_dict)

    def get_logits_from_input_dict(self, q_dicts, bsize=32, progress_bar=True):
        self.model.eval()
        result_logits = []
        iter_count = (len(q_dicts) - 1) // bsize + 1
        ranger = range(iter_count) if not progress_bar else tqdm.trange(iter_count)
        for i in ranger:
            l = self.get_logits_from_input_dict_(q_dicts[i*bsize:(i+1) * bsize]).detach().cpu().numpy().tolist()
            result_logits.extend(l)
        return np.array(result_logits)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


def resize(sent_A, sent_B, max_length, t5tok):
    combined_cap = max_length - 30
    toks_A = t5tok(sent_A)['input_ids']
    toks_B = t5tok(sent_B)['input_ids']
    
    toks_A_new, toks_B_new = [], []
    total_token_count = 0
    for i in range(max(len(toks_A), len(toks_B)) - 1):
        if total_token_count >= combined_cap:
            break
        if i < len(toks_A) - 1:
            toks_A_new.append(toks_A[i])
            total_token_count += 1
        
        if total_token_count >= combined_cap:
            break
        if i < len(toks_B) - 1:
            toks_B_new.append(toks_B[i])
            total_token_count += 1
    new_A, new_B = t5tok.decode(toks_A_new), t5tok.decode(toks_B_new)
    return new_A, new_B


def query_paired_fitness_controlled_(h, pos, neg, num_examples, m, max_length=128):
    t5tok = m.tokenizer
    q = 'Is it true that compared to sentence B, sentence A ' + h + '?'
    
    pairs = []
    for i in range(num_examples):
        sent_A = random.choice(pos)
        sent_B = random.choice(neg)
        pairs.append((sent_A, sent_B))

    qc_dicts = []
    for sent_A, sent_B in pairs:
        sent_A, sent_B = resize(sent_A, sent_B, max_length, t5tok)
        c = 'sentence A: ' + sent_A + '\n\nsentence B: ' + sent_B
        qc_dicts.append({'q': q, 'c': c})
    positive_logits = m.get_logits_from_input_dict(qc_dicts, bsize=BSIZE)
    pos_score = np.mean((np.e ** positive_logits[:,1]) > 0.5)

    qc_dicts = []

    for sent_A, sent_B in pairs:
        sent_A, sent_B = resize(sent_A, sent_B, max_length, t5tok)
        c = 'sentence A: ' + sent_B + '\n\nsentence B: ' + sent_A
        qc_dicts.append({'q': q, 'c': c})
    reverse_logits = m.get_logits_from_input_dict(qc_dicts, bsize=BSIZE)
    reverse_score = np.mean((np.e ** reverse_logits[:,1]) > 0.5)
    return {
        'h_score': pos_score - reverse_score,
        'h': h,
        'dicts': pairs,
        'logits': {
            'positive_logits': positive_logits,
            'reverse_logits': reverse_logits
        }
    }
    
    
def query_single_fitness_controlled_(h, pos, neg, num_examples, m):
    q = 'Is it true that this sentence ' + h + '?'
    pos, neg = list(pos), list(neg)
    random.shuffle(pos)
    random.shuffle(neg)
    
    pos_examples = pos[:num_examples]
    qc_dicts = [{'q': q, 'c': s} for s in pos_examples]
    pos_logits = m.get_logits_from_input_dict(qc_dicts, bsize=BSIZE)[:,1]
    
    neg_examples =  neg[:num_examples]
    qc_dicts = [{'q': q, 'c': s} for s in neg_examples]
    neg_logits = m.get_logits_from_input_dict(qc_dicts, bsize=BSIZE)[:,1]
    
    pos_score = np.mean((np.e ** pos_logits) > 0.5)
    neg_score = np.mean((np.e ** neg_logits) > 0.5)
    
    return {
        'h_score': pos_score - neg_score,
        'h': h,
        'dicts': (pos_examples, neg_examples),
        'logits': {
            'pos_logits': pos_logits,
            'neg_logits': neg_logits
        }
    }


class DummyVerifier:

    def __init__(self):
        self.seq_length = 128
        print('loading verifier')
        self.model = T5ZeroShotClfQA('allenai/unifiedqa-t5-large', 
                                     max_seq_length=self.seq_length, half_precision=True)
        print('verifier loaded')
        self.description = 'Unifiedqa t5-large for debugging'
    
    def return_verification(self, h, pos, neg, num_examples):
        result = query_paired_fitness_controlled_(h, pos, neg, num_examples, self.model, max_length=self.seq_length)
        return result

    
class Verifier0514:

    def __init__(self):
        self.seq_length = 256
        print('loading verifier')
        self.model = T5ZeroShotClfQA('ruiqi-zhong/t5verifier_0514', 
                                     max_seq_length=self.seq_length, half_precision=True)
        print('verifier loaded')
        self.description = 'Similar to Verifier 1207, though the fine-tuned on clean verification data'
    
    def return_verification(self, h, pos, neg, num_examples):
        result = query_paired_fitness_controlled_(h, pos, neg, num_examples, self.model, max_length=self.seq_length)
        return result


class UnifiedQASingle:
    
    def __init__(self):
        self.seq_length = 256
        print('loading verifier')
        self.model = T5ZeroShotClfQA('allenai/unifiedqa-t5-11b', 
                                     max_seq_length=self.seq_length, half_precision=True)
        self.model.eval()
        print('verifier loaded')
        self.description = 'UnifiedQA evaluated on single hypotheses'
    
    def return_verification(self, h, pos, neg, num_examples):
        result = query_single_fitness_controlled_(h, pos, neg, num_examples, self.model)
        return result
    

class UnifiedQA_v2Single:
    
    def __init__(self):
        self.seq_length = 256
        print('loading verifier')
        self.model = T5ZeroShotClfQA('allenai/unifiedqa-v2-t5-11b-1251000',
                                     max_seq_length=self.seq_length, half_precision=True)
        self.model.eval()
        print('verifier loaded')
        self.description = 'UnifiedQA-v2 evaluated on single hypotheses'
    
    def return_verification(self, h, pos, neg, num_examples):
        result = query_single_fitness_controlled_(h, pos, neg, num_examples, self.model)
        return result
    

class UnifiedQA_v2:

    def __init__(self):
        self.seq_length = 256
        print('loading verifier')
        self.model = T5ZeroShotClfQA('allenai/unifiedqa-v2-t5-11b-1251000',
                                     max_seq_length=self.seq_length, half_precision=True)
        self.model.eval()
        print('verifier loaded')
        self.description = 'UnifiedQA-v2 evaluated on comparison hypotheses'
    
    def return_verification(self, h, pos, neg, num_examples):
        result = query_paired_fitness_controlled_(h, pos, neg, num_examples, self.model, max_length=self.seq_length)
        return result
    

def init_verifier(verifier_name):
    return name2verifier_cls[verifier_name]()

    
name2verifier_cls = {
    'ruiqi-zhong/t5verifier_0514': Verifier0514,
    'dummy': DummyVerifier,
    'unifiedqasingle': UnifiedQASingle,
    'unifiedqa_v2single': UnifiedQA_v2Single,
    'unifiedqa_v2': UnifiedQA_v2
}

