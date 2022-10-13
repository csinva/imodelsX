from typing import Iterable, Optional, Tuple

import argparse

import torch
import torch.nn as nn
import transformers

from imodelsx.iprompt.utils import PrefixLoss, PrefixModel


class GumbelPrefixModel(PrefixModel):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_embedding: nn.Parameter

    def __init__(self, args: argparse.Namespace, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, preprefix: str):
        super().__init__(args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=preprefix)
        self.word_weights = nn.Parameter(
            torch.randn((1, args.num_learned_tokens, self.vocab_size)), requires_grad=True
        )
        # TODO: argparse for tau
        # (lower tau -> more spiky)
        self.tau = 10
        # TODO: argparse for tau_anneal
        self.tau_anneal = 1.02

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.word_weights]
    
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        self.tau = self.tau / self.tau_anneal
        print(f"𝛕 = {self.tau:.2f}")

    def embed_input_ids(self, input_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert prefix_ids is None, "cannot provide custom prefix IDs for Gumbel"
        # word_weights_dist = (word_weights * 1000).softmax(dim=-1)
        prefix_embedding_words_dist = nn.functional.gumbel_softmax(
            self.word_weights.repeat((len(input_ids), 1, 1)), tau=self.tau, dim=-1, hard=False
        )
        
        print(
            "trying words:", self.tokenizer.decode(
                prefix_embedding_words_dist[0].argmax(dim=1).tolist()),
            "with prob", prefix_embedding_words_dist[0].max().item()
        )
        prefix_embedding = prefix_embedding_words_dist @ self.token_embedding.weight

        input_ids = torch.cat(
            (prefix_embedding_words_dist.argmax(dim=-1), input_ids), dim=1
        )
        outputs = torch.cat(
            # concatenate prefix + example
            (prefix_embedding, self.token_embedding.forward(input_ids)), dim=1
        )
        return input_ids, outputs
