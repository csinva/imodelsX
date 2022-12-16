from typing import Any, Dict, List, Iterable, Optional, Tuple, Union

import abc
import argparse
import collections
import dataclasses
import functools
import heapq
import random
import transformers
import torch
from torch.utils.data import DataLoader
from torch import nn
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEBUG_VERBOSE = False


def get_token_replacements_single_mask(
    dataloader: DataLoader, model: transformers.AutoModelForMaskedLM,
    tokenizer: transformers.AutoTokenizer, init_prefix_template: str, num_candidates: int)-> List[str]:
    """Given a template like `{mask} the numbers`, returns the `num_candidates` most likely
    single-token replacements for `{mask}` given `model`.
    """
    single_mask_prefix_str = init_prefix_template.format(mask=tokenizer.mask_token)
    all_mask_probs = torch.zeros((tokenizer.vocab_size,), dtype=float).to(device)
    for idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        full_text = [f'{single_mask_prefix_str} {input_text}' for input_text in batch['text']]
        if idx == 0:
            print('Sample input: ', full_text[0])
        inputs = tokenizer(full_text, return_tensors='pt', padding='longest')
        with torch.no_grad():
            outputs = model(**inputs.to(device))
        mask_idxs = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero()
        # TODO: how to do this better in torch?
        mask_probs = outputs.logits[mask_idxs[:, 0], mask_idxs[:, 1]].log_softmax(dim=1)
        all_mask_probs += mask_probs.sum(dim=0)
        
    prefix_idxs = all_mask_probs.topk(num_candidates).indices
    return [init_prefix_template.format(mask=tokenizer.decode(idx)) for idx in prefix_idxs]


def get_prefix_from_mlm(
        dataloader: DataLoader,
        mlm_name: str,
        num_candidates: int,
        template: str
    ) -> List[str]:
    """ Getting prefix from MLM."""
    mlm = transformers.RobertaForMaskedLM.from_pretrained(mlm_name).to(device)
    mlm_tokenizer = transformers.AutoTokenizer.from_pretrained(mlm_name)
    # template = "{mask} the two numbers to get the answer."
    # template = "{mask} the input number to get the answer."
    # template = "Return the{mask} of the input."

    candidates = get_token_replacements_single_mask(
        dataloader=dataloader,
        model=mlm, tokenizer=mlm_tokenizer,
        init_prefix_template=template,
        num_candidates=num_candidates
    )
    mlm.to('cpu') # no need for mlm on GPU anymore
    return candidates


def compute_log_ppl_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Computes LM perplexity loss given logits for next tokens and original input IDs.
    Exponentiate this quantity if you want the actual perplexity.
    """
    # logits gives us the probability of each token that comes after each token in input_ids.
    # so they have the same shape. But we only want to compute ppl using the tokens we have,
    # i.e. not the first true token (which we don't have logits for) or the last predicted token
    # (which we don't know the true id for). so we have to shift each by one index.
    assert logits.shape[0:2] == input_ids.shape
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]

    # now flatten along sequence length so we can compute crossentropy.
    batch_size, sequence_length, vocab_size = logits.shape
    assert input_ids.shape == (batch_size, sequence_length)
    logits = logits.reshape((batch_size * sequence_length, vocab_size))
    input_ids = input_ids.reshape((batch_size * sequence_length, ))
    
    loss = torch.nn.functional.cross_entropy(
        input=logits,
        target=input_ids,
        reduction='mean'
    )
    return loss

@dataclasses.dataclass
class PrefixLoss:
    """Computes next-token-prediction loss with optional language modeling component.
    """
    gamma: float
    tokenizer: transformers.PreTrainedTokenizer # for debugging

    def _compute_fluency_loss(
            self, logits: torch.Tensor, input_ids: torch.Tensor
        ) -> torch.Tensor:
        if self.gamma == 0:
            return torch.tensor(0.0).to(device)
        return compute_log_ppl_loss(logits=logits, input_ids=input_ids)

    def _compute_token_loss(
            self, next_token_logits: torch.Tensor, next_token_idxs: torch.Tensor, answer_mask: torch.Tensor
        ) -> torch.Tensor:
        batch_size, vocab_size = next_token_logits.shape
        assert next_token_idxs.shape == (batch_size,)

        if answer_mask is not None:
            assert answer_mask.shape == (vocab_size,)
            next_token_logits = torch.where(
                answer_mask[None],
                next_token_logits, torch.tensor(float('-inf')).to(device)
            )
                
        return torch.nn.functional.cross_entropy(
            input=next_token_logits,
            target=next_token_idxs,
            reduction='mean'
        )
    
    def __call__(
            self,
            input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            logits: torch.Tensor,
            answer_mask: torch.Tensor,
        ) -> torch.Tensor:
        """Computes loss.

        Args:
            input_ids (int torch.Tensor): array of token IDs for inputs
            next_token_ids (int torch.Tensor): array of token IDs for the word
                that comes after the input
            logits (float torch.Tensor): logits for all output tokens, including
                the next one
            answer_mask (bool torch.Tensor): mask over tokens to remove irrelevant ones

        Returns: float torch.Tensor scalar, loss value (lower is better).
        """
        fluency_loss = (
            self._compute_fluency_loss(
                logits=logits,
                input_ids=input_ids
            )
        )

        token_loss = (
            self._compute_token_loss(
                next_token_logits=logits[:, -1, :],
                next_token_idxs=next_token_ids,
                answer_mask=answer_mask,
            )
        )

        loss = token_loss + (self.gamma * fluency_loss)
        if DEBUG_VERBOSE: 
            print(f">> loss for input string: {self.tokenizer.decode(input_ids[0])}")
            print(f"\tLoss = {loss:.3f}")
        return loss


class PrefixModel(nn.Module, abc.ABC):
    args: argparse.Namespace
    loss_func: PrefixLoss
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    
    def __init__(self, args: argparse.Namespace, loss_func: PrefixLoss, model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer, preprefix: str):
        super().__init__()
        self.args = args
        self.loss_func = loss_func
        self.model = model
        self.tokenizer = tokenizer

    @property
    def id_to_word(self) -> Dict[int, str]:
        # track token-to-word mapping 
        return {num: word for word, num in self.tokenizer.vocab.items()}
    
    @property
    def _is_gpt_neox(self) -> bool:
        return isinstance(self.model, transformers.GPTNeoXModel) or isinstance(self.model, transformers.GPTNeoXForCausalLM)
    
    @property
    def _is_t5(self) -> bool:
        return isinstance(self.model, transformers.T5ForConditionalGeneration)

    @property
    def _is_opt(self) -> bool:
        return isinstance(self.model, transformers.OPTForCausalLM)

    @property
    def transformer(self) -> nn.Module:
        if self._is_gpt_neox:
            return self.model._modules['gpt_neox']
        elif self._is_t5:
            return self.model.encoder
        elif self._is_opt:
            return self.model._modules['model'].decoder
        else:
            return self.model._modules['transformer']

    @property
    def token_embedding(self) -> nn.Embedding:
        if self._is_gpt_neox:
            return self.transformer.embed_in
        elif self._is_opt:
            return self.transformer.embed_tokens
        else:
            return self.transformer.wte
    
    @property
    def vocab_size(self) -> int:
        return self.token_embedding.weight.shape[0] # 50_257 for GPT2

    @property 
    def token_embedding_dim(self) -> int:
        return self.token_embedding.weight.shape[1] # often 768, or 2560 for some larger models
    
    def prepare_batch(self, batch: Dict[str, str]) -> Tuple[str, str]:
        """Preprocesses text from `batch['input']` and `batch['output']` for inputting into prefix model.
        """
        x_text = [f'. {prompt}' for prompt in batch['input']]
        y_text = [answer.rstrip().rstrip('.') for answer in batch['output']] # strip whitespace at the end.
        return x_text, y_text

    def forward(
            self,
            input_ids: torch.Tensor,
            prefix_ids: Optional[torch.Tensor],
        ) -> torch.Tensor:
        new_input_ids, embeddings = self.embed_input_ids(
            input_ids=input_ids, prefix_ids=prefix_ids
        )

        # Automatically set attention mask and position-ids
        attention_mask = ~(new_input_ids == self.tokenizer.pad_token_id)

        assert new_input_ids.shape == embeddings.shape[0:2]
        return new_input_ids, self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
    
    def pre_epoch(self) -> None:
        return
    
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        return
    
    def compute_metrics(self) -> Dict[str, Any]:
        return {}
    
    def serialize(self) -> Dict[str, Any]:
        """Writes stuff to disk after training."""
        return {}

    @abc.abstractproperty
    def trainable_params(self) -> Iterable[nn.Parameter]:
        raise NotImplementedError()

    @abc.abstractmethod
    def embed_input_ids(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """To be implemented by subclasses -- embeds input ids and includes some sort of prefix,
        for example, in the case of prompt-tuning, by prepending a continuous embedding.
        """
        raise NotImplementedError()
    
    def init_continuous_prefix(self, num_tokens: int) -> nn.Parameter:
        return nn.Parameter(
            self.token_embedding.weight.mean(dim=0, keepdim=True)[None].repeat(1, num_tokens, 1), requires_grad=True
        )
    
    def init_discrete_prefix(self, num_tokens: int) -> nn.Parameter:
        if self.args.autoprompt_init_strategy == 'random':
            return torch.randint(low=0, high=self.tokenizer.vocab_size, size=(num_tokens,))
        else:
            start_word_id = torch.tensor([self.tokenizer.vocab['the']], dtype=int)
            # print(f"start_word_id = {start_word_id}")
            return start_word_id.repeat((num_tokens,))

    def _compute_loss_with_set_prefix(
            self,
            original_input_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            possible_answer_mask: torch.Tensor,
            prefix_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # roll tensors together to put padding at the end. slow but I know it's right. (TODO: tensorize)
        input_ids = []
        num_pad_tokens = (original_input_ids == self.tokenizer.eos_token_id).sum(dim=1)
        for i in range(len(original_input_ids)):
            if num_pad_tokens[i] == 0:
                next_tensor = torch.cat((original_input_ids[i], next_token_ids[i]), dim=0)
            else:
                num_non_pad_tokens = original_input_ids.shape[1] - num_pad_tokens[i]
                padding = torch.full(size=(num_pad_tokens[i], ), fill_value=self.tokenizer.eos_token_id).to(device)
                next_tensor = torch.cat((original_input_ids[i][:num_non_pad_tokens], next_token_ids[i], padding), dim=0)
            input_ids.append(
                next_tensor
            )

        input_ids = torch.stack(input_ids)
        assert input_ids.shape == (original_input_ids.shape[0], original_input_ids.shape[1] + next_token_ids.shape[1])

        # feed into the model. prefix-handling is implemented in PrefixModel::forward.
        full_input_ids, outputs = self.forward(
            input_ids=input_ids,
            prefix_ids=prefix_ids,
        )
        # make sure we have same number of predictions as tokens
        assert full_input_ids.shape == outputs.logits.shape[:2]

        # get first predicted token logits
        next_token_idx = (~(full_input_ids == self.tokenizer.eos_token_id)).cumsum(dim=1).argmax(dim=1)
        next_token_logits = outputs.logits[torch.arange(len(original_input_ids)), next_token_idx-1]

        # compute first-token acc
        if possible_answer_mask is None:
            n_correct = (
                next_token_logits.argmax(dim=-1) == next_token_ids[:, 0]
            ).int().sum()
        else:
            # apply possible answer mask for single-token
            next_token_logits = torch.where(
                possible_answer_mask[None],
                next_token_logits, torch.tensor(float('-inf')).to(device)
            )
            n_correct = (
                (next_token_logits.exp() * possible_answer_mask).argmax(dim=-1)
                    ==
                next_token_ids[:, 0]
            ).int().sum()

        # compute loss from first token
        original_losses = torch.nn.functional.cross_entropy(
            input=next_token_logits,
            target=next_token_ids[:, 0],
            ignore_index=self.tokenizer.pad_token_id,
            reduction='none'
        )

        # add loss from other tokens
        b, label_sequence_length = next_token_ids.shape
        if label_sequence_length > 1:
            other_next_token_logits = (
                outputs.logits[:, -label_sequence_length:-1]
                    .reshape((b * (label_sequence_length-1), -1))
            )
            other_next_token_ids = (
                next_token_ids[:, 1:]
                    .reshape((b * (label_sequence_length-1),))
            )
            other_losses = torch.nn.functional.cross_entropy(
                input=other_next_token_logits,
                target=other_next_token_ids,
                ignore_index=self.tokenizer.pad_token_id,
                reduction='none'
            )
            # take the mean of losses on the batch level, and normalize for length
            all_losses = torch.cat(
                (original_losses[:, None], other_losses.reshape((b, -1))), dim=1
            )
            num_tokens_per_output = (
                ~(next_token_ids == self.tokenizer.bos_token_id)).sum(dim=1)
            all_losses = all_losses.sum(dim=1) / num_tokens_per_output
            assert all_losses.shape == (b,)
            loss = all_losses.mean()
        else:
            loss = original_losses.mean()
        
        if DEBUG_VERBOSE: 
            print(f">> loss for input string: {self.tokenizer.decode(full_input_ids[0])}")
            print(f"\tLoss = {loss:.3f}")

        return full_input_ids, loss, n_correct
    
    def compute_loss_and_call_backward(
            self,
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            possible_answer_mask: Optional[torch.Tensor],
            full_text_tokenized: Optional[transformers.BatchEncoding] = None
        ) -> Tuple[torch.Tensor, int]:
        """Computes loss using `self.loss_func`.
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        original_input_ids = x_tokenized.input_ids
        next_token_ids = y_tokenized.input_ids[:, 0] # only compute loss over next token

        input_ids, outputs = self.forward(input_ids=original_input_ids, prefix_ids=None)

        next_token_logits = outputs.logits[:, -1, :]

        n_correct = (
            next_token_logits.argmax(dim=-1)
                ==
            next_token_ids
        ).int().sum()

        loss = self.loss_func(
            input_ids=input_ids,
            next_token_ids=next_token_ids,
            logits=outputs['logits'],
            answer_mask=possible_answer_mask
        )
        loss.backward()
        return loss, n_correct
    
    def check_early_stop(self) -> bool:
        """Allow prefix models to stop early."""
        return False


def mean(_list: List[Union[int, float]]) -> float:
    return sum(_list) / len(_list)


class PrefixPool:
    """Tracks a pool of candidate prefixes and their associated metrics over time."""
    criterion: str
    tokenizer: transformers.PreTrainedTokenizer
    # 
    _all_losses: Dict[Tuple[int], List[float]]
    _avg_loss: Dict[Tuple[int], float]
    _all_accuracy: Dict[Tuple[int], List[float]]
    _avg_accuracy: Dict[Tuple[int], float]
    _best_prefix_by_start_token: Dict[int, Tuple[Tuple[int], float]]

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, criterion: str):
        self.tokenizer = tokenizer
        self.criterion = criterion
        # tuple (input_ids) -> float (loss)
        self._avg_loss = {}
        self._all_losses = collections.defaultdict(list)
        # tuple (input_ids) -> int (n_correct)
        self._avg_accuracy = {}
        self._all_accuracy = collections.defaultdict(list)
        # 
        self._best_prefix_by_start_token = {}
        # 
        self._topk_strategy = 'different_start_token' # ['different_start_token', 'all']
    
    @property
    def prefixes(self) -> List[Tuple[int]]:
        return self._avg_loss.keys()
    
    @property
    def num_start_tokens(self) -> int:
        """Number of different start tokens seen across all prefixes."""
        return len(self._best_prefix_by_start_token.keys())
    
    def print(self, topk: int, min_occurrences: int = 2) -> None:
        top_token_ids = self.topk(k=topk, min_occurrences=min_occurrences)
        ########################### Debugging code ##########################
        # import pandas as pd
        # vd = pd.DataFrame(self._avg_loss.items(), columns=['prefix', 'loss'])
        # vd['prefix_str'] = vd['prefix'].map(self.tokenizer.decode)
        # vd['n'] = vd['prefix'].map(lambda p: len(self._all_losses[p]))
        # vd.sort_values(by='loss')["prefix_str"].iloc[:25]
        # vd.sort_values(by=['n', 'loss'], ascending=[False, True])[["n", "prefix_str"]].iloc[:25]
        #####################################################################
        if not len(top_token_ids): return
        print((" " * 45), ("*" * 20), "Population", ("*" * 20))
        for token_ids in top_token_ids:
            prefix_str = "{:>65}".format(self.tokenizer.decode(list(token_ids)).replace("\n", "\\\\n"))
            loss_str = f"{self._avg_loss[token_ids]:.3f}"
            acc_str = f"{self._avg_accuracy[token_ids]*100:.1f}"
            print(prefix_str, "\t\t", loss_str, "\t\t", acc_str)
        print()
    
    def initialize_prefix(self, prefix: torch.Tensor):
        prefix = tuple(prefix.cpu().tolist())
        self._avg_loss[prefix] = 10_000.0
        self._avg_accuracy[prefix] = 0
        self._best_prefix_by_start_token.setdefault(prefix[0], (prefix, (10_000.0,)))

    def topk(self, *args, **kwargs) -> List[Tuple[int]]:
        if self._topk_strategy == 'different_start_token':
            return self.topk_with_different_start_token(*args, **kwargs)
        elif self._topk_strategy == 'all':
            return self.topk_all(*args, **kwargs)
        else:
            raise ValueError(f'Unknown strategy {self._topk_strategy}')

    def topk_with_different_start_token(
        self,
        k: int,
        min_occurrences: Optional[int] = None
        ) -> List[Tuple[int]]:
        all_prefixes = [p for p, score in self._best_prefix_by_start_token.values()]
        top_prefixes = self._topk_from_prefixes(
            all_prefixes, k=k, min_occurrences=min_occurrences
        )
        if not len(top_prefixes):
            # get top prefixes the first time
            top_prefixes = self._topk_from_prefixes(
                all_prefixes, k=k, min_occurrences=0
            )
        n_so_far = len(top_prefixes)
        if n_so_far < k:
            # fallback if we don't have enough first-tokens yet
            # more_prefixes = (
            #     set(self.topk_all(k=k, min_occurrences=min_occurrences))
            #     - set(top_prefixes)
            # )
            num_prefixes_to_add = k - len(top_prefixes)
            # num_prefixes_to_add = min(len(more_prefixes), num_prefixes_to_add)
            more_prefixes = [
                random.choice(top_prefixes) for _ in range(num_prefixes_to_add)
            ]
            top_prefixes += more_prefixes
        top_prefixes.sort(key=self._score)
        return top_prefixes

    def topk_all(self, k: int, min_occurrences: Optional[int] = None) -> List[Tuple[int]]:
        all_prefixes = self._avg_loss.keys()
        return self._topk_from_prefixes(
            all_prefixes, k=k, min_occurrences=min_occurrences
        )
    
    def _score(self, prefix: Tuple[int]) -> Tuple[float]:
        criterion = self.criterion
        if criterion == 'loss':
            # sort by min loss
            return (self._avg_loss[prefix], )
        elif criterion == 'combined':
            return (-1 * round(self._avg_accuracy[prefix], 2), self._avg_loss[prefix])
        else:
            return (-1 * self._avg_accuracy[prefix], 2)
    
    def _topk_from_prefixes(
        self,
        prefixes: Iterable[Tuple[int]],
        k: int, 
        min_occurrences: Optional[int] = None
        ) -> List[Tuple[int]]:
        if min_occurrences:
            prefixes = {
                prefix for prefix in prefixes
                if len(self._all_accuracy[prefix]) > min_occurrences
            }

        population = [(self._score(p), p) for p in prefixes]
        topk_pop = heapq.nsmallest(k, population)
        topk_pop.sort(key = lambda t: t[0])
        return [prefix_ids for _, prefix_ids in topk_pop]

    def update(self, prefix: torch.Tensor, loss: torch.Tensor, accuracy: torch.Tensor):
        # todo abstract these data strcutures into a class
        prefix = tuple(prefix.cpu().flatten().tolist())
        self._all_losses[prefix].append(loss.item())
        self._avg_loss[prefix] = mean(self._all_losses[prefix])
        self._all_accuracy[prefix].append(accuracy.item())
        self._avg_accuracy[prefix] = mean(self._all_accuracy[prefix])

        # track best score for each starting token
        self._best_prefix_by_start_token.setdefault(prefix[0], (prefix, (1000.0,)))
        score = self._score(prefix)
        best_prefix, best_score = self._best_prefix_by_start_token[prefix[0]]
        if score < best_score:
            self._best_prefix_by_start_token[prefix[0]] = (prefix, score)
    
    def __len__(self) -> int:
        return len(self._avg_loss)
