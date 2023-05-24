from typing import Iterable, Optional, Tuple

import argparse
import collections
import os
import random
import pickle

import torch
import torch.nn as nn
import tqdm
import transformers

from .utils import device, PrefixLoss, PrefixModel


VERBOSE = False # whether to print grads, etc.
TOP_K = 20 # for printing grads, etc.


class HotFlip(PrefixModel):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_ids: torch.Tensor
    prefix_embedding: nn.Parameter
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
        # HotFlip-specific parameters.
        self._min_loss = float('inf')
        self._num_tokens = args.num_learned_tokens # TODO argparse for n_tokens
        self._num_candidates_per_prefix_token = args.hotflip_num_candidates # TODO argparse for this too
        self._swap_token_idx = 0

        self._tested_prefix_ids = collections.defaultdict(lambda: 0)
        # Sort both a version with a preprefix ("The function to compute is") and a version
        # where the full prefix is discovered by HotFlip without any assistance.
        preprefix_ids = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else []
        if preprefix:
            preprefix_ids.extend(self.tokenizer.encode(preprefix))
        self.preprefix_ids = torch.tensor(preprefix_ids, dtype=int).to(device)
        self.prefix_ids = None
        self._set_prefix_ids(
            self.init_discrete_prefix(num_tokens=self._num_tokens)
        )
        print(f"preprefix: '{preprefix}'")

        # disable grads to model
        for p in self.model.parameters(): p.requires_grad = False

        # track data specific to HotFlip
        self._epoch = 0
        self._data = []
        self._loss_for_prefix = {}
        # 
        self.prefix_before_input = args.prefix_before_input

    def check_early_stop(self) -> bool:
        """Allow prefix models to stop early."""
        if self.args.early_stopping_steps == -1:
            return False
        return self._steps_since_new_prefix >= self.args.early_stopping_steps
    
    def _set_prefix_ids(self, new_ids: torch.Tensor) -> None:
        assert new_ids.ndim == 1, "cannot set prefix with more than 1 dim (need list of IDs)"

        # Track steps since new prefix to enable early stopping
        if (self.prefix_ids is not None) and (self.prefix_ids == new_ids).all():
            self._steps_since_new_prefix += 1
        else:
            self._steps_since_new_prefix = 0
        

        self.prefix_ids = new_ids.to(device)
        self.prefix_embedding = nn.Parameter(
            self.token_embedding.to(device).forward(self.prefix_ids), requires_grad=True
        )
        # track prefixes we've tried
        self._tested_prefix_ids[(tuple(new_ids.flatten().tolist()), self._swap_token_idx)] += 1

    def pre_epoch(self) -> None:
        # Print closest tokens at the beginning of each epoch.
        if VERBOSE:
            print("*" *  30)
            print(f"Epoch {epoch}. Closest tokens to '{prefix_str}':")
            word_distances =  ((self.token_embedding.weight - self.prefix_embedding.reshape(1, emb_dim))**2).sum(1)
            assert word_distances.shape == (50_257,)
            topk_closest_words = word_distances.topk(k=TOP_K, largest=False)
            for _id, _dist in zip(topk_closest_words.indices.cpu().tolist(), topk_closest_words.values.cpu().tolist()):
                print(f'\t{self.id_to_word[_id]} ({_id}): {_dist:.3f}')
            print("*" * 30)
    
    @property
    def _prefix_token_grad(self) -> torch.Tensor:
        """Gradient of the prefix tokens wrt the token embedding matrix."""
        return torch.einsum('nd,vd->nv', self.prefix_embedding.grad, self.token_embedding.weight)
    
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

        _input_ids, loss, n_correct = self._compute_loss_with_set_prefix(
            original_input_ids=original_input_ids,
            next_token_ids=next_token_ids, # only compute loss over next token
            possible_answer_mask=possible_answer_mask
        )

        loss.backward()

        # self._set_prefix_ids(best_prefix)
        return loss, n_correct
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        token_idx = self._swap_token_idx
        token_grads = self._prefix_token_grad
        top_tokens_per_position = (
            token_grads.topk(k=self._num_candidates_per_prefix_token, dim=1, largest=False).indices
        )
        assert top_tokens_per_position.shape == (self._num_tokens, self._num_candidates_per_prefix_token)

        top_swap_tokens = top_tokens_per_position[token_idx, :]
        #
        # Get most likely tokens.
        #
        prefix_until_swap_ids = torch.cat(
            (self.preprefix_ids.to(device), self.prefix_ids[:token_idx].to(device)), dim=0
        )[None].to(device)
        with torch.no_grad():
            all_preprefix_logits = self.model(prefix_until_swap_ids)
            swap_token_logits = all_preprefix_logits.logits[:, -1, :]

        rvocab = {v: k for k,v in self.tokenizer.vocab.items()}
        # dist_sum = (swap_token_logits.log_softmax(dim=1) * .7 + (-1 * token_grads).log_softmax(dim=1))
        # for v in (swap_token_logits.log_softmax(dim=1) * .7 + (-1 * token_grads).log_softmax(dim=1)).topk(10).indices.flatten(): print(rvocab[v.item()])

        alpha = 0.0 # TODO argparse for this alpha
        print(f"HotFlip alpha = {alpha}")
        token_losses = (
            (swap_token_logits.log_softmax(dim=1) * alpha + (-1 * token_grads).log_softmax(dim=1))
        )
        top_swap_tokens = token_losses.argsort(descending=True).flatten()

        # if we've already tried this (prefix, swap_token_idx) combo, then let's try the next n candidates.
        _n = self._tested_prefix_ids[tuple(self.prefix_ids.flatten().tolist()), token_idx] - 1
        assert _n >= 0, "something went wrong"
        top_swap_tokens = top_swap_tokens[(_n * self._num_candidates_per_prefix_token) : (_n+1) * self._num_candidates_per_prefix_token]
        # 
        # Evaluate candidates.
        # 
        all_candidate_losses = torch.zeros(self._num_candidates_per_prefix_token, dtype=float).to(device)
        all_n_correct = torch.zeros(self._num_candidates_per_prefix_token, dtype=int).to(device)
        best_loss = self._min_loss

        mask = torch.nn.functional.one_hot(
            torch.tensor(token_idx), num_classes=self._num_tokens
        ).bool().to(device)

        # Evaluate each prefix.
        for batch in tqdm.tqdm(dataloader, desc='evaluating HotFlip candidates', colour='red', leave=False):
            # Loop in this order so we only tokenize each thing once.
            x_text, y_text = self.prepare_batch(batch=batch)
            input_ids = self.tokenizer(x_text, return_tensors='pt', padding='longest')['input_ids'].to(device)
            next_token_ids = self.tokenizer(y_text, return_tensors='pt', padding='longest')['input_ids'].to(device)
            # only evaluate on single next-token
            next_token_ids = next_token_ids[:, 0]
            for candidate_idx in range(self._num_candidates_per_prefix_token):
                new_token_id = top_swap_tokens[candidate_idx]
                prefix_ids = torch.where(
                    mask, new_token_id, self.prefix_ids.to(device)
                ).to(device)
                with torch.no_grad():
                    _input_ids, loss, n_correct = (
                        self._compute_loss_with_set_prefix(
                            original_input_ids=input_ids,
                            next_token_ids=next_token_ids,
                            possible_answer_mask=possible_answer_mask,
                            prefix_ids=prefix_ids
                        )
                    )
                all_candidate_losses[candidate_idx] += loss
                all_n_correct[candidate_idx] += n_correct

        ##################################################################################################################
        hotflip_out_path = os.path.join(self.args.save_dir_unique, 'hotflip_grads_data.p')
        for _i in range(self._num_candidates_per_prefix_token):
            token_id = top_swap_tokens[_i].item()
            # rank, prefix, token_id, token_grad, loss_with_this_token, n_correct_with_this_token
            self._data.append(
                (_i, self.prefix_ids.tolist(), token_id, token_grads.flatten()[token_id].item(), all_candidate_losses[_i].item(), all_n_correct[_i].item())
            )
        pickle.dump(self._data, open(hotflip_out_path, 'wb'))
        ##################################################################################################################

        #
        # Collect losses for all prefixes. Then set prefix to best one we haven't seen before.
        #
        for candidate_idx in range(self._num_candidates_per_prefix_token):
            new_token_id = top_swap_tokens[candidate_idx]
            prefix_ids = tuple(
                torch.where(
                    mask, new_token_id, self.prefix_ids.to(device)
                ).tolist()
            )
            self._loss_for_prefix[prefix_ids] = (
                all_candidate_losses[candidate_idx].item(),
                all_n_correct[candidate_idx].item()
            )
        
        # next prefix is the one we know about with the min loss that we haven't tried
        # so far.
        best_prefix_ids = min(self._loss_for_prefix, key=lambda p: self._loss_for_prefix.get(p)[0])
        best_loss, best_n_correct =  self._loss_for_prefix[best_prefix_ids]

        # if loss < self._min_loss:
        #     self._min_loss = loss
        #     best_prefix_ids = prefix_ids

        # 
        # Pick top candidate and reset self._min_loss. (TODO: Support beam width > 1.)
        # 
        old_prefix_str = self.tokenizer.decode(self.preprefix_ids.tolist() + self.prefix_ids.tolist())
        new_prefix_str = self.tokenizer.decode(self.preprefix_ids.tolist() + list(best_prefix_ids))
        print(f'[Loss = {best_loss/len(dataloader):.2f}] // Old prefix: {old_prefix_str} // New prefix: {new_prefix_str} // New n_correct = {best_n_correct}')

        self._swap_token_idx = (self._swap_token_idx + 1) % self._num_tokens
        # self._swap_token_idx = random.randint(0, (self._num_tokens-1))

        self._set_prefix_ids(torch.tensor(best_prefix_ids))

        return

    @property
    def prefix_embedding_token_ids(self) -> torch.Tensor:
        return self.prefix_embedding.argmax(dim=-1)

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.prefix_embedding]

    def embed_input_ids(
        self, input_ids: torch.Tensor, next_token_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets token embeddings for tokens given by `input_ids` prefixed by `prefix_ids`.

        If not provided, `prefix_ids` is replaced with `self.prefix_ids`
        at every position.

        Args:
            input_ids (int torch.Tensor) -- IDs for batch of sentences
            prefix_ids (Optional int torch.Tensor) -- IDs for a single prefix
                to be prepended before each input ID. If not provided,
                will be overridden with prefix from `self.prefix_ids`.

        Returns:
            input_ids (int torch.Tensor) -- IDs of all tokens, including prefix
            outputs (float torch.Tensor): embedded tokens
        """
        batch_size = len(input_ids)
        if prefix_ids is None:
            prefix_ids = self.prefix_ids
            prefix_embedding = self.prefix_embedding
            
        else:
            prefix_embedding = self.token_embedding.forward(prefix_ids)

        # concatenate preprefix (fixed) + prefix (learned) + example
        prefix_ids = prefix_ids[None].to(device).repeat((batch_size, 1)).to(device)
        preprefix_ids = self.preprefix_ids[None].to(device).repeat((batch_size, 1)).to(device)

        if self.prefix_before_input:
            full_input_ids = torch.cat(
                (preprefix_ids, prefix_ids, input_ids, next_token_ids), dim=1
            )
            outputs = torch.cat(
                (
                    self.token_embedding.forward(preprefix_ids),
                    prefix_embedding[None].repeat((batch_size, 1, 1)),
                    self.token_embedding.forward(input_ids),
                    self.token_embedding.forward(next_token_ids),
                ), dim=1
            )
        else:
            full_input_ids = torch.cat(
                (input_ids, preprefix_ids, prefix_ids, next_token_ids), dim=1
            )
            outputs = torch.cat(
                (
                    self.token_embedding.forward(input_ids),
                    self.token_embedding.forward(preprefix_ids),
                    prefix_embedding[None].repeat((batch_size, 1, 1)),
                    self.token_embedding.forward(next_token_ids),
                ), dim=1
            )
        return full_input_ids, outputs
