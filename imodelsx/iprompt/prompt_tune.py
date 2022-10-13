from typing import Any, Dict, Iterable, Optional, Tuple

import argparse

import torch
import torch.nn as nn
import transformers

from imodelsx.iprompt.utils import PrefixLoss, PrefixModel


class PromptTunedModel(PrefixModel):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    prefix_embedding: nn.Parameter
    def __init__(self, args: argparse.Namespace, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, preprefix: str):
        super().__init__(args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=preprefix)
        self.prefix_embedding = self.init_continuous_prefix(num_tokens=args.num_learned_tokens)

    def embed_input_ids(self, input_ids: torch.Tensor, prefix_ids: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert prefix_ids is None, "cannot provide custom prefix IDs for prompt-tuning"
        token_embeddings = self.token_embedding.forward(input_ids)
        return None, torch.cat(
            (self.prefix_embedding.repeat((len(input_ids), 1, 1)), token_embeddings), dim=1
        )

    @property
    def trainable_params(self) -> Iterable[nn.Parameter]:
        return [self.prefix_embedding]
    
    def serialize(self):
        save_dir = self.args.save_dir_unique
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.prefix_embedding, open(os.path.join(save_dir, 'prefix_embedding.p'), 'wb'))

    def compute_metrics(self) -> Dict[str, Any]:
        return {
            'embs': self.prefix_embedding.detach().cpu().numpy(),
            'grads': self.prefix_embedding.grad.detach().cpu().numpy(),
        }
