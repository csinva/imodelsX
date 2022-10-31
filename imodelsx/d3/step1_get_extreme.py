import json
import os
import pickle as pkl
import random
from itertools import chain
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
import torch
import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch import nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrain_model = 'roberta-large'


class RoBERTaSeq(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrain_model)
    
    def forward(self, **inputs):
        model_outputs = self.model(**inputs)
        model_output_dict = vars(model_outputs)
        
        seq_lengths = torch.sum(inputs['attention_mask'], dim=-1).detach().cpu().numpy()
        model_output_dict['highlight'] = [[1./seq_length for _ in range(seq_length)] for seq_length in seq_lengths]
        return model_output_dict
        

class RoBERTaSeqAttn(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(pretrain_model)
        self.clf_layer = nn.Linear(self.model.config.hidden_size, 2)
        self.attn_layer = nn.Linear(self.model.config.hidden_size, 1)
        self.sm = nn.Softmax(dim=-1)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.loss_func = nn.NLLLoss()
        
    
    def forward(self, **inputs):
        last_hidden_state = self.model(input_ids=inputs['input_ids'], 
                                   attention_mask=inputs['attention_mask']).last_hidden_state
        
        attn_logits = self.attn_layer(last_hidden_state).squeeze(axis=-1)
        attn_logits[inputs['attention_mask'] == 0] = float('-inf')
        attention = self.sm(attn_logits)
        
        seq_lengths = torch.sum(inputs['attention_mask'], dim=-1).detach().cpu().numpy()
        aggregated_repr = torch.einsum('bs,bsh->bh', attention, last_hidden_state)
        
        logits = self.lsm(self.clf_layer(aggregated_repr))
        
        return_dict = {
            'logits': logits,
            'highlight': [attention[i][:s].detach().cpu().numpy() for i, s in enumerate(seq_lengths)]
        }
        if 'labels' in inputs:
            loss = self.loss_func(logits, inputs['labels'])
            return_dict['loss'] = loss
        return return_dict
        

lsm = torch.nn.LogSoftmax(dim=-1)
def cv(pos, neg, K):
    return [
        {
            'train_pos': [p for i, p in enumerate(pos) if i % K != k],
            'train_neg': [n for i, n in enumerate(neg) if i % K != k],
            'test_pos': [p for i, p in enumerate(pos) if i % K == k],
            'test_neg': [n for i, n in enumerate(neg) if i % K == k],
        }
        for k in range(K)
    ]


def get_spans(tok, text):
    be = tok(text)
    length = len(be['input_ids'])
    results = []
    for i in range(length):
        if i in (0, length - 1):
            results.append((0, 0))
        else:
            start, end = be.token_to_chars(i)
            results.append((start, end))
    return results


def train_and_eval(cv_dict, num_steps=2000, batch_size=16):
    max_length = 128
    train_data_dicts = list(chain(
        [{'input': x, 'label': 1} for x in cv_dict['train_pos']], 
        [{'input': x, 'label': 0} for x in cv_dict['train_neg']], 
    ))
    # model = RobertaForSequenceClassification.from_pretrained(pretrain_model).to(device)
    # model = RoBERTaSeq().to(device)
    model = RoBERTaSeqAttn().to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 400, num_steps)
    tok = AutoTokenizer.from_pretrained(pretrain_model)
    
    for step in tqdm.trange(num_steps):
        random.shuffle(train_data_dicts)
        input_texts = [d['input'] for d in train_data_dicts[:batch_size]]
        inputs = tok(input_texts, return_tensors='pt', truncation=True, max_length=max_length, padding=True).to(device)
        labels = torch.tensor([d['label'] for d in train_data_dicts[:batch_size]]).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        loss.backward()
        if step % 2 == 1:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    def evaluate(texts):
        all_logits, all_highlights = [], []
        cur_start = 0
        while cur_start < len(texts):
            texts_ = texts[cur_start:cur_start + batch_size]
            inputs = tok(texts_, return_tensors='pt', truncation=True, max_length=max_length, padding=True).to(device)
            model_output_dict = model(**inputs)
            logits = lsm(model_output_dict['logits'].detach().cpu()).numpy().tolist()
            all_highlights.extend(model_output_dict['highlight'])
            all_logits.extend(logits)
            cur_start += batch_size
        assert len(all_logits) == len(texts)
        
        all_spans = [get_spans(tok, text) for text in texts]
        assert len(all_spans) == len(all_highlights)
        for a, b in zip(all_spans, all_highlights):
            assert len(a) == len(b) or len(a) >= max_length
        
        highlights = [
            {s: h for s, h in zip(spans, highlights) if s != (0, 0)} 
            for spans, highlights in zip(all_spans, all_highlights)
        ]
        
        return {
            'logits': np.array(all_logits),
            'highlights': highlights
        }
    
    pos_eval_dict = evaluate(cv_dict['test_pos'])
    pos_logits, pos_highlights = pos_eval_dict['logits'][:,1], pos_eval_dict['highlights']
    
    neg_eval_dict = evaluate(cv_dict['test_neg'])
    neg_logits, neg_highlights = neg_eval_dict['logits'][:,0], neg_eval_dict['highlights']
    
    return {
        'test_pos_scores': pos_logits,
        'test_neg_scores': neg_logits,
        'test_pos_highlight': pos_highlights,
        'test_neg_highlight': neg_highlights 
    }

def return_extreme_values(pos, neg, num_steps=2000, num_folds=4, batch_size=16):
    pos2score, neg2score = {}, {}
    pos2highlight, neg2highlight = {}, {}
    
    for fold_idx, cv_dict in enumerate(cv(pos, neg, num_folds)):
        print('fold', fold_idx + 1, '/', num_folds)
        test_scores = train_and_eval(cv_dict, num_steps, batch_size)
        for pos_text, score, highlight in zip(cv_dict['test_pos'], test_scores['test_pos_scores'], test_scores['test_pos_highlight']):
            pos2score[pos_text] = score
            pos2highlight[pos_text] = highlight
        for neg_text, score, highlight in zip(cv_dict['test_neg'], test_scores['test_neg_scores'], test_scores['test_neg_highlight']):
            neg2score[neg_text] = score
            neg2highlight[neg_text] = highlight
    return {
        'pos2score': pos2score,
        'neg2score': neg2score,
        'pos2highlight': pos2highlight,
        'neg2highlight': neg2highlight
    }


    
        
    