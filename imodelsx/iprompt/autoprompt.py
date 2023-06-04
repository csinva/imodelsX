from typing import Any, Dict, List, Optional, Tuple

import argparse
import functools
import os
import pickle
import random

import pandas as pd
import torch
import tqdm
import transformers

from .hotflip import HotFlip
from .utils import device, PrefixLoss, PrefixModel, PrefixPool


class AutoPrompt(HotFlip):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: torch.nn.Parameter
    preprefix: str

    def __init__(
        self,
        args: argparse.Namespace,
        loss_func: PrefixLoss,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        preprefix: str = ''
    ):
        super().__init__(
            args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=preprefix
        )
        self._do_final_reranking = args.iprompt_do_final_reranking
        # AutoPrompt-specific parameters.
        self._num_candidates_per_prefix_token = 32  # V_cand in autoprompt paper
        # This helps us know which were the best prefixes to return over time
        self._prefix_pool = PrefixPool(
            tokenizer=self.tokenizer,
            criterion='loss'  # in ['loss', 'acc', 'combined']
        )
        self._autoprompt_verbose = True
        self._num_min_occurrences = 1
        # Will rank and save this many prefixes at the end of training.
        self._num_prefixes_to_test = 64
    
    def test_prefixes(
        self,
        prefixes: List[Tuple[int]], 
        eval_dataloader: torch.utils.data.DataLoader, 
        possible_answer_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes loss & accuracy for each prefix on data in dataloader. Used to rank
        prefixes at the end of training.
        """
        all_candidate_losses = torch.zeros(len(prefixes), dtype=torch.float32)
        all_candidate_n_correct = torch.zeros(
            len(prefixes), dtype=torch.float32)
        total_n = 0
        for batch in tqdm.tqdm(eval_dataloader, desc=f'evaluating {len(prefixes)} prefixes'):
            if (self.args.n_shots > 1) and (self.args.single_shot_loss): ##
               batch['input'] = batch['last_input'] ##
            x_text, y_text = self.prepare_batch(batch=batch)
            tok = functools.partial(
                self.tokenizer, return_tensors='pt', padding='longest',
                truncation=True, max_length=self.args.max_length  # TODO set max_length on self
            )
            x_tokenized = tok(x_text).to(device)
            y_tokenized = tok(y_text).to(device)
            total_n += len(x_tokenized.input_ids)

            next_token_ids = y_tokenized.input_ids
            for i in range(len(prefixes)):
                with torch.no_grad():
                    _cand_input_ids, cand_loss, cand_n_correct = (
                        self._compute_loss_with_set_prefix(
                            original_input_ids=x_tokenized.input_ids,
                            next_token_ids=next_token_ids,
                            possible_answer_mask=possible_answer_mask,
                            prefix_ids=torch.tensor(prefixes[i]).to(device),
                        )
                    )
                all_candidate_losses[i] += cand_loss.item()
                all_candidate_n_correct[i] += cand_n_correct.item()
        return all_candidate_losses.cpu().tolist(), (all_candidate_n_correct / total_n).cpu().tolist()

    def serialize(self, eval_dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> Dict[str, Any]:
        """Writes stuff to disk. Saves other stuff to save as full results file.
        """

        # Uncomment following lines to save all the prefixes we tested.
        # save_dir = self.args.save_dir_unique
        # os.makedirs(save_dir, exist_ok=True)
        # pickle.dump(self._prefix_pool, open(os.path.join(save_dir, 'prefix_pool.p'), 'wb'))

        all_prefixes = self._prefix_pool.topk_all(
            k=self._num_prefixes_to_test, min_occurrences=3)

        if not len(all_prefixes):
            # In the case where we get no prefixes here (i.e. prompt generation
            # only ran for a single step) just take anything from prefix pool.
            all_prefixes = random.choices(list(self._prefix_pool.prefixes), k=self._num_prefixes_to_test)

        if self._do_final_reranking:
            all_losses, all_accuracies = self.test_prefixes(
                prefixes=all_prefixes,
                eval_dataloader=eval_dataloader,
                possible_answer_mask=possible_answer_mask
            )
            df = pd.DataFrame(
                zip(*[all_prefixes, all_losses, all_accuracies]),
                columns=['prefix', 'loss', 'accuracy']
            )
            df = df.sort_values(by=['accuracy', 'loss'], ascending=[
                                False, True]).reset_index()
        else:
            all_prefixes = list(self._prefix_pool.prefixes)
            all_losses = [self._prefix_pool._avg_loss.get(p, -1) for p in all_prefixes]
            all_accuracies = [self._prefix_pool._avg_accuracy.get(p, -1) for p in all_prefixes]

            df = pd.DataFrame(
                zip(*[all_prefixes, all_losses, all_accuracies]),
                columns=['prefix', 'loss', 'accuracy']
            )
        df = df.sort_values(by='accuracy', ascending=False).reset_index()

        df['prefix_str'] = df['prefix'].map(self.tokenizer.decode)
        df['n_queries'] = df['prefix'].map(
            lambda p_ids: len(self._prefix_pool._all_losses[p_ids]))

        print('Final prefixes')
        print(df.head())

        return {
            "prefix_ids": df['prefix'].tolist(),
            "prefixes": df['prefix_str'].tolist(),
            "prefix_train_acc": df['accuracy'].tolist(),
            "prefix_train_loss": df['loss'].tolist(),
            "prefix_n_queries": df['n_queries'].tolist(),
        }

    def compute_loss_and_call_backward(
        self,
        x_tokenized: transformers.BatchEncoding,
        y_tokenized: transformers.BatchEncoding,
        possible_answer_mask: torch.Tensor,
        full_text_tokenized: Optional[transformers.BatchEncoding] = None
    ) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.

        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        original_input_ids = x_tokenized.input_ids
        next_token_ids = y_tokenized.input_ids  # only compute loss over next token

        current_input_ids, current_loss, current_n_correct = self._compute_loss_with_set_prefix(
            original_input_ids=original_input_ids,
            next_token_ids=next_token_ids,
            possible_answer_mask=possible_answer_mask,
            prefix_ids=None,
        )
        current_loss.backward()

        self._autoprompt_verbose: print(
            f'** {self.tokenizer.decode(self.prefix_ids)}: {current_loss:.2f}')
        
        # track running accuracy of this prefix.
        self._prefix_pool.update(
            prefix=self.prefix_ids,
            loss=current_loss,
            accuracy=(current_n_correct/len(original_input_ids))
        )

        # print an update.
        self._prefix_pool.print(topk=10, min_occurrences=1)

        #
        # Get top token replacements
        #
        token_grads = self._prefix_token_grad
        if self._is_t5:
            # t5 has extra vocab tokens for no reason:
            # https://github.com/huggingface/transformers/issues/4875#issuecomment-647634437
            assert token_grads.shape == (
                self._num_tokens, len(self.tokenizer.vocab) + 28
            )
            token_grads = token_grads[:, :-28]
        assert token_grads.shape == (
            self._num_tokens, len(self.tokenizer.vocab))
        top_tokens_per_position = (
            token_grads.topk(
                k=self._num_candidates_per_prefix_token, dim=1, largest=False).indices
        )
        assert top_tokens_per_position.shape == (
            self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[self._swap_token_idx, :]
        #
        # Get most likely tokens.
        #
        top_swap_tokens = token_grads.argsort(descending=False).flatten()
        top_swap_tokens = top_swap_tokens[0:
                                          self._num_candidates_per_prefix_token]

        # rank candidates
        mask = torch.nn.functional.one_hot(
            torch.tensor(self._swap_token_idx), num_classes=self._num_tokens
        ).bool().to(device)
        candidate_prefix_ids = torch.where(
            mask, top_swap_tokens[:, None], self.prefix_ids[None, :])
        is_current_prefix_mask = (
            candidate_prefix_ids == self.prefix_ids).all(dim=1)
        candidate_prefix_ids = candidate_prefix_ids[~is_current_prefix_mask]

        # get best prefix
        num_candidates = len(candidate_prefix_ids)
        all_candidate_losses = torch.zeros(
            num_candidates, dtype=float).to(device)
        all_n_correct = torch.zeros(num_candidates, dtype=int).to(device)
        for i in range(num_candidates):
            with torch.no_grad():
                cand_input_ids, cand_loss, cand_n_correct = (
                    self._compute_loss_with_set_prefix(
                        original_input_ids=original_input_ids,
                        next_token_ids=next_token_ids,
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=candidate_prefix_ids[i],
                    )
                )
            all_candidate_losses[i] = cand_loss
            all_n_correct[i] = cand_n_correct

            # self._autoprompt_verbose: print(
                # f'** \t{self.tokenizer.decode(candidate_prefix_ids[i])}: {cand_loss:.2f}')

            self._prefix_pool.update(
                prefix=candidate_prefix_ids[i],
                loss=cand_loss,
                accuracy=(cand_n_correct / len(original_input_ids))
            )

        # randomly change the token to swap
        self._swap_token_idx = random.randint(0, (self._num_tokens-1))
        # get best prefix we've seen
        if all_candidate_losses.min() < current_loss:
            best_prefix = candidate_prefix_ids[all_candidate_losses.argmin()]
            best_prefix_loss = all_candidate_losses.min()
            best_prefix_n_correct = all_n_correct[all_candidate_losses.argmin(
            )]
            if self._autoprompt_verbose:
                print("** set new prefix", best_prefix)
        else:
            best_prefix = self.prefix_ids
            best_prefix_loss = current_loss
            best_prefix_n_correct = current_n_correct
            if self._autoprompt_verbose:
                print("** set same prefix", best_prefix)

        self._set_prefix_ids(best_prefix)
        return best_prefix_loss, best_prefix_n_correct

    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        #
        # Get candidate IDs for every position.
        #
        pass
