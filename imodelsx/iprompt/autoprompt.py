from typing import Any, Dict, Optional, Tuple

import argparse
import os
import pickle
import random
import torch
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
        # AutoPrompt-specific parameters.
        self._num_candidates_per_prefix_token = 32 # V_cand in autoprompt paper
        # This helps us know which were the best prefixes to return over time
        self._prefix_pool = PrefixPool(
            tokenizer=self.tokenizer,
            criterion='loss'  # in ['loss', 'acc', 'combined']
        )
        self._VERBOSE = False
        self._num_min_occurrences = 1
    
    def serialize(self) -> Dict[str, Any]:
        """Writes stuff to disk. Saves other stuff to save as full results file.
        """
        save_dir = self.args.save_dir_unique
        os.makedirs(save_dir, exist_ok=True)

        # Uncomment following line to save all the prefixes we tested.
        # pickle.dump(self._prefix_pool, open(os.path.join(save_dir, 'prefix_pool.p'), 'wb'))
        
        N_p = 64 # num prefixes to save
        
        # logic here is that we want to see a sample a good number of times before
        # we actually have a good estimate of its loss.
        num_min_occurrences = self._num_min_occurrences

        topk_prefixes = self._prefix_pool.topk_all(k=N_p, min_occurrences=num_min_occurrences)
        topk_different_prefixes = self._prefix_pool.topk_with_different_start_token(k=N_p, min_occurrences=num_min_occurrences)
        top_prefixes = topk_prefixes + topk_different_prefixes
        top_prefix_types = ((["topk_all"] * len(topk_prefixes)) + (["topk_with_different_start_token"] * len(topk_different_prefixes)))
        top_prefixes_str = [self.tokenizer.decode(p) for p in top_prefixes]
        top_prefix_accs = [self._prefix_pool._avg_accuracy[p] for p in top_prefixes]
        top_prefix_losses = [self._prefix_pool._avg_loss[p] for p in top_prefixes]
        top_prefix_n_queries = [len(self._prefix_pool._all_losses[p]) for p in top_prefixes]
        return {
            "prefix_ids": top_prefixes,
            "prefix_type": top_prefix_types,
            "prefixes": top_prefixes_str,
            "prefix_train_acc": top_prefix_accs,
            "prefix_train_loss": top_prefix_losses,
            "prefix_n_queries": top_prefix_n_queries,
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
        next_token_ids = y_tokenized.input_ids # only compute loss over next token

        current_input_ids, current_loss, current_n_correct = self._compute_loss_with_set_prefix(
            original_input_ids=original_input_ids,
            next_token_ids=next_token_ids,
            possible_answer_mask=possible_answer_mask,
            prefix_ids=None,
        )
        current_loss.backward()

        self._VERBOSE: print(f'** {self.tokenizer.decode(self.prefix_ids)}: {current_loss:.2f}')
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
        assert token_grads.shape == (self._num_tokens, len(self.tokenizer.vocab))
        top_tokens_per_position = (
            token_grads.topk(k=self._num_candidates_per_prefix_token, dim=1, largest=False).indices
        )
        assert top_tokens_per_position.shape == (self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[self._swap_token_idx, :]
        #
        # Get most likely tokens.
        #
        top_swap_tokens = token_grads.argsort(descending=False).flatten()
        top_swap_tokens = top_swap_tokens[0:self._num_candidates_per_prefix_token]

        # rank candidates
        mask = torch.nn.functional.one_hot(
            torch.tensor(self._swap_token_idx), num_classes=self._num_tokens
        ).bool().to(device)
        candidate_prefix_ids = torch.where(mask, top_swap_tokens[:, None], self.prefix_ids[None, :])
        is_current_prefix_mask = (candidate_prefix_ids == self.prefix_ids).all(dim=1)
        candidate_prefix_ids = candidate_prefix_ids[~is_current_prefix_mask]

        # get best prefix
        num_candidates = len(candidate_prefix_ids)
        all_candidate_losses = torch.zeros(num_candidates, dtype=float).to(device)
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

            self._VERBOSE: print(f'** \t{self.tokenizer.decode(candidate_prefix_ids[i])}: {cand_loss:.2f}')

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
            best_prefix_n_correct = all_n_correct[all_candidate_losses.argmin()]
            if self._VERBOSE: print("** set new prefix", best_prefix)
        else:
            best_prefix = self.prefix_ids
            best_prefix_loss = current_loss
            best_prefix_n_correct = current_n_correct
            if self._VERBOSE: print("** set same prefix", best_prefix)


        self._set_prefix_ids(best_prefix)
        return best_prefix_loss, best_prefix_n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        pass