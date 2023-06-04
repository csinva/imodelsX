from typing import Callable, Dict, List, Tuple

import datasets
import functools
import os
import random
import string
import numpy as np
import time
import torch
import transformers
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
from imodelsx.iprompt import (
    AutoPrompt, iPrompt,
    PrefixLoss, PrefixModel,
    PromptTunedModel, HotFlip, GumbelPrefixModel
)
from imodelsx.iprompt.llm import get_llm
import pandas as pd
import logging
import pickle as pkl
from torch.utils.data import DataLoader
from datetime import datetime


"""
Explaining Patterns in Data with Language Models via Interpretable Autoprompting

Chandan Singh*, John X. Morris*, Jyoti Aneja, Alexander M. Rush, Jianfeng Gao
https://arxiv.org/abs/2210.01848
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_cls_dict = {
    'autoprompt': AutoPrompt,
    'iprompt': iPrompt,
    'gumbel': GumbelPrefixModel,
    'hotflip': HotFlip,
    'prompt_tune': PromptTunedModel,
}

def get_prompts_api(
        data: List[str], 
        llm: Callable,
        prompt_template: str, 
    ):
    data_str = random.choice(data)
    prompt = prompt_template(data=data_str).strip()
    answer = llm(prompt, max_new_tokens=24)
    return [answer]


def run_iprompt_api(
    r: Dict[str, List],
    input_strs: List[str],
    output_strs: List[str],
    model: PrefixModel,
    tokenizer: transformers.PreTrainedTokenizer,
    llm_api: str,
    save_dir: str = 'results',
    lr: float = 1e-4,
    batch_size: int = 64,
    max_length: int = 128,
    n_epochs: int = 100,
    n_shots: int = 1,
    single_shot_loss: bool = True,
    accum_grad_over_epoch: bool = False,
    max_n_datapoints: int = 10**4,
    max_n_steps: int = 10**4,
    epoch_save_interval: int = 1,
    mask_possible_answers: bool = False,
    verbose: int = 0,
):
    """
    Trains a model, either by optimizing continuous embeddings or finding an optimal discrete embedding.

    Params
    ------
    r: dict
        dictionary of things to save
    """


    # remove periods and newlines from the output so we actually use the tokens
    # for the reranking step in iPrompt
    output_strs = [s.rstrip().rstrip('.') for s in output_strs]

    r['train_start_time'] = time.time()
    model.train()

    logging.info("beginning iPrompt with n_shots = %d", n_shots)

    assert len(input_strs) == len(
        output_strs), "input and output must be same length to create input-output pairs"
    text_strs = list(map(lambda t: f'{t[0]}{t[1]}.', zip(input_strs, output_strs)))
    df = pd.DataFrame.from_dict({
        'input': input_strs,
        'output': output_strs,
        'text': text_strs,
    })
    if n_shots > 1:
        d2 = defaultdict(list)
        for i in range(max_n_datapoints):
            all_shots = df.sample(n=n_shots, replace=False)
            d2['text'].append('\n\n'.join(all_shots['text'].values))
            #
            last_input = all_shots.tail(n=1)['input'].values[0]
            d2['input'].append(
                ''.join(all_shots['text'].values[:-1]) + last_input)
            d2['last_input'].append(last_input)
            #
            last_output = all_shots.tail(n=1)['output'].values[0]
            d2['output'].append(last_output)
            #
        df = pd.DataFrame.from_dict(d2)
    
    # shuffle rows
    if max_n_datapoints < len(df):
        df = df.sample(n=max_n_datapoints, replace=False)
    dset = datasets.Dataset.from_pandas(df)
    dset.shuffle()
    print(f'iPrompt got {len(dset)} datapoints, now loading model...')

    model = model.to(device)
    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=True, drop_last=False)

    prompt_template = "{prompt_start}\n\n{data}\n\n{prompt_end}"
    prompt_template = functools.partial(
        prompt_template.format,
        prompt_start=model.llm_candidate_regeneration_prompt_start,
        prompt_end=model.llm_candidate_regeneration_prompt_end,
    )
    
    prompts = []
    # "gpt-3.5-turbo", "text-curie-001"
    llm = get_llm( 
        checkpoint=llm_api, role="user")
    stopping_early = False
    total_n = 0
    total_n_steps = 0
    total_n_datapoints = 0
    for epoch in range(n_epochs):       
        print(f'Beginning epoch {epoch}')
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in pbar:
            total_n_steps += 1
            if (n_shots > 1) and (single_shot_loss):
                batch['input'] = batch['last_input']
            x_text, y_text = model.prepare_batch(batch=batch)

            tok = functools.partial(
                model.tokenizer, return_tensors='pt', padding='longest',
                truncation=True, max_length=max_length)
            text_tokenized = tok(batch['text']).to(device)
            text_detokenized = model.tokenizer.batch_decode(
                text_tokenized['input_ids'], 
                skip_special_tokens=True,
            )
            
            prompts.extend(
                get_prompts_api(
                    data=text_detokenized, 
                    llm=llm, 
                    prompt_template=prompt_template, 
                )
            )

            total_n += len(x_text)
            total_n_datapoints += len(x_text)
            if (total_n_datapoints > max_n_datapoints) or (total_n_steps > max_n_steps):
                stopping_early = True
                break

        if stopping_early:
            print(f"Ending epoch {epoch} early...")

        # save stuff
        for key, val in model.compute_metrics().items():
            r[key].append(val)

        # r['losses'].append(avg_loss)
        if epoch % epoch_save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

        # Early stopping, check after epoch
        if stopping_early:
            print(
                f"Stopping early after {total_n_steps} steps and {total_n_datapoints} datapoints")
            break

    
    # 
    #   Evaluate model on prefixes
    # 

    # Compute loss only over possible answers to make task easier
    possible_answer_ids = []
    for batch in dataloader:
        y_text = [answer for answer in batch['output']]
        y_tokenized = tokenizer(y_text, return_tensors='pt', padding='longest')
        # only test on the single next token
        true_next_token_ids = y_tokenized['input_ids'][:, 0]
        possible_answer_ids.extend(true_next_token_ids.tolist())
    
    possible_answer_ids = torch.tensor(possible_answer_ids)
    vocab_size = len(tokenizer.vocab)
    possible_answer_mask = (
            torch.arange(start=0, end=vocab_size)[:, None]
            ==
            possible_answer_ids[None, :]
        ).any(dim=1).to(device)
    n_eval = 256
    eval_dset = datasets.Dataset.from_dict(dset[:n_eval])
    eval_dataloader = DataLoader(
        eval_dset, batch_size=batch_size, shuffle=True, drop_last=False)   
    all_prefixes = model.tokenizer(
        [f" {prompt.strip()}" for prompt in prompts], 
        truncation=False, 
        padding=False,
    )["input_ids"]
    all_losses, all_accuracies = model.test_prefixes(
        prefixes=all_prefixes,
        eval_dataloader=eval_dataloader,
        possible_answer_mask=possible_answer_mask
    )

    # 
    #   Store prefix info 
    # 
    df = pd.DataFrame(
        zip(*[all_prefixes, all_losses, all_accuracies]),
        columns=['prefix', 'loss', 'accuracy']
    )
    df = df.sort_values(by=['accuracy', 'loss'], ascending=[
                        False, True]).reset_index()
    df = df.sort_values(by='accuracy', ascending=False).reset_index()

    df['prefix_str'] = df['prefix'].map(
        functools.partial(model.tokenizer.decode, skip_special_tokens=True)
    )

    print('Final prefixes')
    print(df.head())
    r.update({
            "prefix_ids": df['prefix'].tolist(),
            "prefixes": df['prefix_str'].tolist(),
            "prefix_train_acc": df['accuracy'].tolist(),
            "prefix_train_loss": df['loss'].tolist(),
        })

    r['train_end_time'] = time.time()
    r['train_time_elapsed'] = r['train_end_time'] - r['train_start_time']

    pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

    return r


def run_iprompt_local(
    r: Dict[str, List],
    input_strs: List[str],
    output_strs: List[str],
    model: PrefixModel,
    tokenizer: transformers.PreTrainedTokenizer,
    save_dir: str = 'results',
    lr: float = 1e-4,
    batch_size: int = 64,
    max_length: int = 128,
    n_epochs: int = 100,
    n_shots: int = 1,
    single_shot_loss: bool = True,
    accum_grad_over_epoch: bool = False,
    max_n_datapoints: int = 10**4,
    max_n_steps: int = 10**4,
    epoch_save_interval: int = 1,
    mask_possible_answers: bool = False,
    verbose: int = 0,
):
    """
    Trains a model, either by optimizing continuous embeddings or finding an optimal discrete embedding.

    Params
    ------
    r: dict
        dictionary of things to save
    """

    # remove periods and newlines from the output so we actually use the tokens
    # for the reranking step in iPrompt
    output_strs = [s.rstrip().rstrip('.') for s in output_strs]

    r['train_start_time'] = time.time()
    model.train()

    assert len(input_strs) == len(
        output_strs), "input and output must be same length to create input-output pairs"
    text_strs = list(map(lambda t: f'{t[0]}{t[1]}.', zip(input_strs, output_strs)))
    df = pd.DataFrame.from_dict({
        'input': input_strs,
        'output': output_strs,
        'text': text_strs,
    })
    if n_shots > 1:
        d2 = defaultdict(list)
        for i in range(max_n_datapoints):
            all_shots = df.sample(n=n_shots, replace=False)
            d2['text'].append('\n\n'.join(all_shots['text'].values))
            #
            last_input = all_shots.tail(n=1)['input'].values[0]
            d2['input'].append(
                ''.join(all_shots['text'].values[:-1]) + last_input)
            d2['last_input'].append(last_input)
            #
            last_output = all_shots.tail(n=1)['output'].values[0]
            d2['output'].append(last_output)
            #
        df = pd.DataFrame.from_dict(d2)
    # shuffle rows
    if max_n_datapoints < len(df):
        df = df.sample(n=max_n_datapoints, replace=False)
    dset = datasets.Dataset.from_pandas(df)
    dset.shuffle()
    print(f'iPrompt got {len(dset)} datapoints, now loading model...')

    model = model.to(device)
    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=True, drop_last=False)

    # optimizer
    optim = torch.optim.AdamW(model.trainable_params, lr=lr)

    assert model.training

    # Compute loss only over possible answers to make task easier
    possible_answer_ids = []
    for batch in dataloader:
        y_text = [answer for answer in batch['output']]
        y_tokenized = tokenizer(y_text, return_tensors='pt', padding='longest')
        # only test on the single next token
        true_next_token_ids = y_tokenized['input_ids'][:, 0]
        possible_answer_ids.extend(true_next_token_ids.tolist())

    possible_answer_ids = torch.tensor(possible_answer_ids)
    num_unique_answers = len(set(possible_answer_ids.tolist()))
    assert num_unique_answers > 0, "need multiple answers for multiple choice"
    random_acc = 1 / num_unique_answers * 100.0
    majority_count = (
        possible_answer_ids[:, None] == possible_answer_ids[None, :]).sum(dim=1).max()
    majority_acc = majority_count * 100.0 / len(possible_answer_ids)
    print(
        f"Training with {num_unique_answers} possible answers / random acc {random_acc:.1f}% / majority acc {majority_acc:.1f}%")

    vocab_size = len(tokenizer.vocab)

    if mask_possible_answers:
        possible_answer_mask = (
            torch.arange(start=0, end=vocab_size)[:, None]
            ==
            possible_answer_ids[None, :]
        ).any(dim=1).to(device)
    else:
        possible_answer_mask = None

    stopping_early = False
    total_n_steps = 0
    total_n_datapoints = 0
    for epoch in range(n_epochs):
        model.pre_epoch()

        all_losses = []

        total_n = 0
        total_n_correct = 0
        print(f'Beginning epoch {epoch}')
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, batch in pbar:
            total_n_steps += 1
            if (n_shots > 1) and (single_shot_loss):
                batch['input'] = batch['last_input']
            x_text, y_text = model.prepare_batch(batch=batch)

            tok = functools.partial(
                model.tokenizer, return_tensors='pt', padding='longest',
                truncation=True, max_length=max_length)
            x_tokenized = tok(x_text).to(device)
            y_tokenized = tok(y_text).to(device)
            full_text_tokenized = tok(batch['text']).to(device)

            loss, n_correct = model.compute_loss_and_call_backward(
                x_tokenized=x_tokenized,
                y_tokenized=y_tokenized,
                possible_answer_mask=possible_answer_mask,
                full_text_tokenized=full_text_tokenized,
            )

            r["all_losses"].append(loss)
            r["all_n_correct"].append(n_correct)

            total_n += len(x_text)
            total_n_datapoints += len(x_text)
            total_n_correct += n_correct

            all_losses.append(loss)
            pbar.set_description(f"Loss = {loss:.3f}")

            if not accum_grad_over_epoch:
                # if hotflip, autoprompt, etc., grad will be zero
                optim.step()
                optim.zero_grad()

            # Early stopping, check after step
            model_check_early_stop = model.check_early_stop()
            if model_check_early_stop:
                print("model_check_early_stop returned true")
            if (total_n_datapoints > max_n_datapoints) or (total_n_steps > max_n_steps) or model_check_early_stop:
                stopping_early = True
                break

        if stopping_early:
            print(f"Ending epoch {epoch} early...")
        avg_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch}. average loss = {avg_loss:.3f} / {total_n_correct} / {total_n} correct ({total_n_correct/total_n*100:.2f}%)")

        # save stuff
        for key, val in model.compute_metrics().items():
            r[key].append(val)

        # r['losses'].append(avg_loss)
        if epoch % epoch_save_interval == 0:
            os.makedirs(save_dir, exist_ok=True)
            pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

        model.post_epoch(dataloader=dataloader,
                         possible_answer_mask=possible_answer_mask)

        if accum_grad_over_epoch:
            optim.step()
            optim.zero_grad()

        # Early stopping, check after epoch
        if stopping_early:
            print(
                f"Stopping early after {total_n_steps} steps and {total_n_datapoints} datapoints")
            break

    # Serialize model-specific stuff (prefixes & losses for autoprompt, embeddings for prompt tuning, etc.)
    n_eval = 256
    eval_dset = datasets.Dataset.from_dict(dset[:n_eval])
    eval_dataloader = DataLoader(
        eval_dset, batch_size=batch_size, shuffle=True, drop_last=False)
    r.update(model.serialize(eval_dataloader, possible_answer_mask))
    # r.update(model.serialize())

    # save whether prefixes fit the template
    """
    if "prefixes" in r:
        r["prefixes__check_answer_func"] = list(
            map(check_answer_func, r["prefixes"]))
    """

    r['train_end_time'] = time.time()
    r['train_time_elapsed'] = r['train_end_time'] - r['train_start_time']

    pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))

    return r


def eval_model_with_set_prefix(
    dataloader: DataLoader,
    model: PrefixModel,
) -> Tuple[float, float]:
    """
    Evaluates a model based on set prefix. May be called multiple times with different prefixes

    Params
    ------
    r: dict
        dictionary of things to save

    Returns: Tuple[float, float]
        average loss, accuracy per sample over eval dataset
    """
    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc='evaluating data', colour='red', leave=False)
    total_loss = 0.0
    total_n = 0
    total_n_correct = 0
    for idx, batch in pbar:
        x_text, y_text = model.prepare_batch(batch=batch)

        tok = functools.partial(
            model.tokenizer, return_tensors='pt', padding='longest')
        x_tokenized = tok(x_text).to(device)
        y_tokenized = tok(y_text).to(device)
        # full_text_tokenized = tok(batch['text']).to(device)

        with torch.no_grad():
            _input_ids, loss, n_correct = model._compute_loss_with_set_prefix(
                original_input_ids=x_tokenized.input_ids,
                next_token_ids=y_tokenized.input_ids,
                possible_answer_mask=None,  # TODO: implement eval verbalizer
                prefix_ids=None,
            )

        total_loss += loss.item()
        total_n += len(x_text)
        total_n_correct += n_correct

        pbar.set_description(
            f"Acc = {total_n_correct}/{total_n} {(total_n_correct/total_n*100):.2f}%")

    return (total_loss / total_n), (total_n_correct / total_n)


def eval_model(
    r: Dict[str, List],
    dset: datasets.Dataset,
    model: PrefixModel,
    batch_size: int = 500,
    save_dir: str = 'results',
):
    """
    Evaluates a model based on the learned prefix(es).

    Params
    ------
    r: dict
        dictionary of things to save
    """
    r["test_start_time"] = time.time()
    model.eval()
    dataloader = DataLoader(
        dset, batch_size=batch_size, shuffle=False, drop_last=False)

    if r["prefixes"]:
        # if we specified multiple prefixes (autoprompt or iprompt), let's evaluate them all!
        for prefix_ids in tqdm(r["prefix_ids"], desc="evaluating prefixes"):
            model._set_prefix_ids(new_ids=torch.tensor(prefix_ids).to(device))

            loss, acc = eval_model_with_set_prefix(dataloader, model)

            r["prefix_test_loss"].append(loss)
            r["prefix_test_acc"].append(acc)
        r["num_prefixes_used_for_test"] = len(r["prefixes"])

    else:
        # otherwise, there's just one prefix (like for prompt tuning) so just run single eval loop.
        loss, acc = eval_model_with_set_prefix(dataloader, model)
        r["prefix_test_acc"] = loss
        r["prefix_test_loss"] = acc
        r["num_prefixes_used_for_test"] = 1

    r["test_end_time"] = time.time()
    r["test_time_elapsed"] = r["test_end_time"] - r["test_start_time"]
    pkl.dump(r, open(os.path.join(save_dir, 'results.pkl'), 'wb'))
    return r


def explain_dataset_iprompt(
    input_strings: List[str],
    output_strings: List[str],
    checkpoint: str='EleutherAI/gpt-j-6B',
    generation_checkpoint: str = '',
    num_learned_tokens=1,
    save_dir: str = './results',
    lr: float = 0.01,
    pop_size: int = 8,
    pop_criterion: str = 'loss',
    pop_topk_strategy: str = 'different_start_token',
    num_mutations: int = 4,
    prefix_before_input: bool = True,
    do_final_reranking: bool = False,
    num_random_generations: int = 4,
    generation_repetition_penalty: float = 2.0,
    generation_temp: float = 1.0,
    generation_top_p: float = 1.0,
    early_stopping_steps: int = -1,
    llm_float16=False,
    gamma: float = 0.0,
    batch_size: int = 64,
    max_length: int = 128,
    n_epochs: int = 100,
    n_shots: int = 1,
    preprefix: str = '',
    single_shot_loss: bool = True,
    accum_grad_over_epoch: bool = False,
    max_n_datapoints: int = 10**4,
    max_n_steps: int = 10**4,
    epoch_save_interval: int = 1,
    mask_possible_answers: bool = False,
    model_cls: str = 'iprompt',
    lm: transformers.PreTrainedModel = None,
    llm_candidate_regeneration_prompt_start: str = 'Data:',
    llm_candidate_regeneration_prompt_end: str = 'Prompt:',
    verbose: int = 0,  # verbosity level (0 for minimal)
    seed: int = 42,
    llm_api: str = "",
) -> Tuple[List[str], Dict]:
    """Explain the relationship between the input strings and the output strings

    Parameters
    ----------
    input_strings: List[str]
        list of input strings (e.g. "2 + 2")
    output_strings: List[str]
        list of output strings (e.g. "4")
    checkpoint: str
        name of model checkpoint to prompt (e.g. EleutherAI/gpt-j-6B)
    generation_checkpoint: str
        name of model to generate prompts, *only if if different from checkpoint
        used for prompting*. defaults to '' (same model for both).
    prefix_before_input: bool
        whether to prompt the LLM with the prefix before or after the input data
    do_final_reranking: bool
        optionally rerank top prefixes using a single batch. helps prevent
        noisy prefixes from being top at the end, especially when run over a
        small number of iterations or with small batch size.
    generation_temp: float
        temperature for sampling from LLM (defaults to 1.0)
    generation_top_p: float
        p for sampling from LLM, if using top-p sampling (defaults to 1.0, no sampling)
    num_learned_tokens: int
        number of tokens to learn in prompt
    save_dir: str
        directory to save results
    lr: float
        learning rate for prompt tuning
    pop_size: int
        number of prompt candidates to evaluate for each iteration of iprompt
    pop_criterion: str
        criterion for getting top prefixes from prefix pool, in ['loss', 'acc']
    pop_topk_strategy: str
        strategy for getting new prefixes from prefix pool, in ['different_start_token', 'all']
    num_mutations: int
        number of mutations to apply to each prompt candidate
    lm: transformers.PreTrainedModel
        pre-loaded model (overrides checkpoint)
    max_n_data_points: int
        maximum number of data points to use for training
        if n_shots > 1, this many data_points are created by recombining n_shots number of examples


    Returns
    -------
    best_prompts
        List of the best found prompts
    metadata_dict
        Dictionary of metadata from fitting
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if not prefix_before_input:
        tokenizer.truncation_side = 'left'
    tokenizer.eos_token = tokenizer.eos_token or 0
    tokenizer.pad_token = tokenizer.eos_token

    # load the model (unless already loaded)
    def load_lm(checkpoint, tokenizer, llm_float16):
        if llm_float16:
            if checkpoint == "EleutherAI/gpt-j-6B":
                lm = AutoModelForCausalLM.from_pretrained(
                    checkpoint, output_hidden_states=False, pad_token_id=tokenizer.eos_token_id,
                    revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
            else:
                # (only certain models are pre-float16ed)
                print(f"trying to convert {checkpoint} to float16...")
                lm = transformers.AutoModelForCausalLM.from_pretrained(
                    checkpoint, torch_dtype=torch.float16
                )
                lm = lm.half()
        else:
            lm = AutoModelForCausalLM.from_pretrained(
                checkpoint, output_hidden_states=False, pad_token_id=tokenizer.eos_token_id
            )
        return lm
    if lm is None:
        lm = load_lm(checkpoint, tokenizer, llm_float16)
        
    loss_func = PrefixLoss(gamma=gamma, tokenizer=tokenizer)

    if model_cls == 'iprompt':
        model = iPrompt(
            loss_func=loss_func,
            model=lm,
            tokenizer=tokenizer,
            preprefix_str=preprefix,
            pop_size=pop_size,
            pop_criterion=pop_criterion,
            pop_topk_strategy=pop_topk_strategy,
            num_mutations=num_mutations,
            prefix_before_input=prefix_before_input,
            do_final_reranking=do_final_reranking,
            num_random_generations=num_random_generations,
            generation_repetition_penalty=generation_repetition_penalty,
            generation_temp=generation_temp,
            generation_top_p=generation_top_p,
            early_stopping_steps=early_stopping_steps,
            num_learned_tokens=num_learned_tokens,
            max_length=max_length,
            n_shots=n_shots,
            single_shot_loss=single_shot_loss,
            verbose=verbose,
            llm_float16=llm_float16,
            generation_checkpoint=generation_checkpoint,
            llm_candidate_regeneration_prompt_start=llm_candidate_regeneration_prompt_start,
            llm_candidate_regeneration_prompt_end=llm_candidate_regeneration_prompt_end,
        )
    else:
        pass
        """
        model = model_cls_dict[model_cls](
            args=args,
            loss_func=loss_func, model=lm, tokenizer=tokenizer, preprefix=preprefix
        )
        """
    
    iprompt_local = len(llm_api) == 0
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    r = defaultdict(list)
    if iprompt_local:
        r = run_iprompt_local(
            r=r,
            input_strs=input_strings,
            output_strs=output_strings,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            lr=lr,
            batch_size=batch_size,
            max_length=max_length,
            mask_possible_answers=mask_possible_answers,
            n_epochs=n_epochs,
            n_shots=n_shots,
            single_shot_loss=single_shot_loss,
            accum_grad_over_epoch=accum_grad_over_epoch,
            max_n_datapoints=max_n_datapoints,
            max_n_steps=max_n_steps,
            epoch_save_interval=epoch_save_interval,
            verbose=verbose,
        )
    else:
        r = run_iprompt_api(
            r=r,
            input_strs=input_strings,
            output_strs=output_strings,
            model=model,
            tokenizer=tokenizer,
            save_dir=save_dir,
            lr=lr,
            batch_size=batch_size,
            max_length=max_length,
            mask_possible_answers=mask_possible_answers,
            n_epochs=n_epochs,
            n_shots=n_shots,
            single_shot_loss=single_shot_loss,
            accum_grad_over_epoch=accum_grad_over_epoch,
            max_n_datapoints=max_n_datapoints,
            max_n_steps=max_n_steps,
            epoch_save_interval=epoch_save_interval,
            verbose=verbose,
            llm_api=llm_api,
        )
    model = model.cpu()
    return r['prefixes'], r

    # r = eval_model(args=args, r=r, dset=Dataset.from_dict(dset_test[:128]), model=model, tokenizer=tokenizer)


# python api.py --task_name_list add_two --model_cls iprompt --num_learned_tokens 3 --max_dset_size 100 --max_n_datapoints 100 --early_stopping_steps 5 --max_digit 10 --train_split_frac 0.75 --single_shot_loss 1 --save_dir /home/chansingh/tmp/iprompt --checkpoint EleutherAI/gpt-j-6B --batch_size 64 --n_epochs 20
# python api.py --task_name_list add_two --model_cls iprompt --num_learned_tokens 3 --max_dset_size 5000 --max_n_datapoints 5000 --early_stopping_steps 25 --max_digit 10 --train_split_frac 0.75 --single_shot_loss 1 --save_dir /home/chansingh/tmp/iprompt --checkpoint EleutherAI/gpt-j-6B --batch_size 64 --float16 1
# python api.py --n_epochs 1 --max_n_steps 3 --max_n_datapoints 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_cls', type=str,
                        choices=model_cls_dict.keys(),
                        default='iprompt',
                        help='model type to use for training')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size for training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--n_epochs', type=int, default=2,
                        help='number of epochs for training')
    parser.add_argument('--max_n_steps', type=int, default=10**10,
                        help='max number of steps for training')
    parser.add_argument('--max_n_datapoints', type=int, default=20,  # 10**10,
                        help='max number of datapoints for training')
    parser.add_argument('--train_split_frac', type=float,
                        default=None, help='fraction for train-test split if desired')
    parser.add_argument('--max_dset_size', type=int,
                        default=10**4, help='maximum allowable dataset size')
    parser.add_argument('--early_stopping_steps', type=int, default=-1,
                        help='if > 0, number of steps until stopping early after no improvement')
    parser.add_argument('--max_digit', type=int, default=10,
                        help='maximum value of each digit in summand')
    parser.add_argument('--template_num_init_string', type=int, default=0,
                        help='the number of the manually-specified prefix to be initialize with')
    parser.add_argument('--template_num_task_phrasing', type=int, default=0,
                        help='the number of the manual template for any given task (number of options varies with task')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='directory for saving')
    parser.add_argument('--epoch_save_interval', type=int, default=1,
                        help='interval to save results')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='hparam: weight for language modeling loss')
    parser.add_argument('--task_name', type=str, default='add_two',
                        choices=(data.TASKS.keys() - {'SUFFIX'}),
                        help='name of task')
    parser.add_argument('--task_name_list', nargs="*", default=None,
                        help='names of tasks as list; alternative to passing task_name')
    parser.add_argument('--n_shots', type=int, default=1,
                        help='number of shots in the prompt')
    parser.add_argument('--autoprompt_init_strategy', type=str, default='the',
                        choices=('random', 'the'), help='initialization strategy for discrete tokens')
    parser.add_argument('--max_length', type=int, default=128,
                        help='maximum length for inputs')
    parser.add_argument('--single_shot_loss', type=int, default=0,
                        help='if n_shots==0, load multiple shots but only use one compute loss')
    parser.add_argument('--mask_possible_answers', type=int, default=0,
                        help='only compute loss over possible answer tokens')
    parser.add_argument('--hotflip_num_candidates', type=int, default=10,
                        help='number of candidates to rerank, for hotflip')
    parser.add_argument('--accum_grad_over_epoch', type=int, default=0, choices=(0, 1),
                        help='should we clear gradients after a batch, or only at the end of the epoch?')
    parser.add_argument('--num_learned_tokens', type=int, default=1,
                        help='number of learned prefix tokens (for gumbel, hotflip, autoprompt, prompt-tuning)')
    parser.add_argument('--use_preprefix', type=int, default=1, choices=(0, 1),
                        help='whether to use a template pre-prefix')
    parser.add_argument('--iprompt_preprefix_str', type=str, default='',
                        help='Text like "Output the number that" or "Answer F/M if"...'
                        )
    parser.add_argument('--iprompt_pop_size', type=int, default=8,)
    parser.add_argument('--iprompt_num_mutations', type=int, default=4)
    parser.add_argument('--iprompt_num_random_generations',
                        type=int, default=4)
    parser.add_argument('--iprompt_generation_repetition_penalty', type=float, default=2.0,
                        help='repetition penalty for iprompt generations')
    parser.add_argument('--llm_float16', '--float16', '--parsimonious', type=int, default=0, choices=(0, 1),
                        help='if true, loads LLM in fp16 and at low-ram')
    parser.add_argument('--checkpoint', type=str, default="gpt2",
                        choices=(
                            ############################
                            "EleutherAI/gpt-neo-125M",
                            "EleutherAI/gpt-neo-1.3B",
                            "EleutherAI/gpt-neo-2.7B",
                            ############################
                            "EleutherAI/gpt-j-6B",
                            ############################
                            "EleutherAI/gpt-neox-20b",
                            ############################
                            "gpt2",        # 117M params
                            "gpt2-medium",  # 355M params
                            "gpt2-large",  # 774M params
                            "gpt2-xl",     # 1.5B params
                            ############################
                        ),
                        help='model checkpoint to use'
                        )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)

    args.use_generic_query = 0

    if (args.mask_possible_answers) and (args.train_split_frac is not None):
        print("Warning: mask possible answers not supported for eval")

    # iterate over tasks
    if args.task_name_list is not None:
        logging.info('using task_name_list ' + str(args.task_name_list))
    else:
        args.task_name_list = [args.task_name]
    for task_idx, task_name in enumerate(args.task_name_list):
        print(f'*** Executing task {task_idx+1}/{len(args.task_name_list)}')
        # actually set the task
        args.task_name = task_name

        r = defaultdict(list)
        r.update(vars(args))
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO)

        logger.info('loading data and model...')
        # set up saving
        save_dir_unique = datetime.now().strftime("%b_%d_%H_%M_") + \
            ''.join(random.choices(string.ascii_lowercase, k=12))
        save_dir = os.path.join(args.save_dir, save_dir_unique)
        logging.info('saving to ' + save_dir)
        args.save_dir_unique = save_dir

        # get data
        # import this here so it's not needed for the package....
        import iprompt.data as data
        dset, _, _ = data.get_data(
            task_name=args.task_name, n_shots=args.n_shots, train_split_frac=args.train_split_frac, max_dset_size=args.max_dset_size,
            template_num_task_phrasing=args.template_num_task_phrasing, max_digit=args.max_digit
        )
        # pd.DataFrame.from_dict({
        #     'input_strings': dset['input'],
        #     'output_strings': [repr(x) for x in dset['output']],
        # }).to_csv('add_two.csv', index=False)

        prompts, meta = explain_dataset_iprompt(
            input_strings=dset['input'],
            output_strings=dset['output'],
            checkpoint=args.checkpoint,
            save_dir=args.save_dir,
            lr=args.lr,
            pop_size=args.iprompt_pop_size,
            num_mutations=args.iprompt_num_mutations,
            num_random_generations=args.iprompt_num_random_generations,
            generation_repetition_penalty=args.iprompt_generation_repetition_penalty,
            early_stopping_steps=args.early_stopping_steps,
            num_learned_tokens=args.num_learned_tokens,
            llm_float16=args.llm_float16,
            gamma=args.gamma,
            batch_size=args.batch_size,
            max_length=args.max_length,
            n_epochs=args.n_epochs,
            n_shots=args.n_shots,
            single_shot_loss=args.single_shot_loss,
            accum_grad_over_epoch=args.accum_grad_over_epoch,
            max_n_datapoints=args.max_n_datapoints,
            max_n_steps=args.max_n_steps,
            epoch_save_interval=args.epoch_save_interval,
            mask_possible_answers=args.mask_possible_answers,
            model_cls=args.model_cls,
        )
        print('prompts', prompts)
        print('\nmeta', meta)
