from typing import Any, Dict, Iterable, List, Optional, Tuple

import argparse
import collections
import os
import random

import torch
import transformers

from imodelsx.iprompt.autoprompt import AutoPrompt
from imodelsx.iprompt.hotflip import HotFlip
from imodelsx.iprompt.utils import device, PrefixLoss, PrefixModel, PrefixPool


"""
Explaining Patterns in Data with Language Models via Interpretable Autoprompting

Chandan Singh*, John X. Morris*, Jyoti Aneja, Alexander M. Rush, Jianfeng Gao
https://arxiv.org/abs/2210.01848
"""


class iPrompt(AutoPrompt):
    def __init__(
        self,
        loss_func: PrefixLoss,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        preprefix_str: str = '',
        prefix_before_input: bool = True,
        pop_criterion: str = 'loss',
        pop_topk_strategy: str = 'different_start_token',
        pop_size: int = 8,
        num_mutations: int = 4,
        num_random_generations: int = 4,
        generation_repetition_penalty: float = 2.0,
        generation_temp: float = 1.0,
        generation_top_p: float = 1.0,
        do_final_reranking: bool = False,
        early_stopping_steps: int = -1,
        num_learned_tokens: int = 1,
        max_length: int = 128,
        verbose: int = 0,
        llm_float16: bool = True,
        generation_checkpoint: str = '',
        n_shots: int = 1,
        single_shot_loss: bool = True,
        llm_candidate_regeneration_prompt_start: str = 'Data:',
        llm_candidate_regeneration_prompt_end: str = 'Prompt:',
    ):
        args = argparse.Namespace()
        args.prefix_before_input = prefix_before_input
        args.num_learned_tokens = num_learned_tokens
        args.hotflip_num_candidates = None
        args.autoprompt_init_strategy = None
        args.save_dir_unique = '.'
        args.n_shots = n_shots
        args.single_shot_loss = single_shot_loss
        args.max_length = max_length
        args.iprompt_do_final_reranking = do_final_reranking
        super().__init__(
            args=args, loss_func=loss_func, model=model, tokenizer=tokenizer, preprefix=''
        )
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens = False
        ####################################################################
        # iPrompt-specific parameters
        self._pop_size = pop_size
        self._topk_pop_sample = (self._pop_size + 4) # sample next population from this num of top things. set higher for more randomness.
        self._num_mutations_per_ex = num_mutations # num mutations for each population item
        self._num_random_generations = num_random_generations # extra random examples to throw in there (won't get mutated)
        self._generation_temp = generation_temp
        self._generation_top_p = generation_top_p
        self._generation_repetition_penalty = generation_repetition_penalty # 1 means no penalty
        self._pop_initialized = False
        self._generation_bad_words_ids = [
            self.tokenizer.encode('\n'),
            self.tokenizer.encode('\n\n'),
            self.tokenizer.encode('\n\n\n')
        ]
        ####################################################################
        self.conditioning_strategy = '' # This arg is only used for ablations.
        self.other_generation_model = None
        if generation_checkpoint:
            self.other_generation_model = load_lm_from_checkpoint(
                generation_checkpoint, float16=llm_float16
            )
        ####################################################################
        self._prefix_pool = PrefixPool(
            tokenizer=self.tokenizer,
            criterion=pop_criterion, # 'loss'  # in ['loss', 'acc', 'combined']
            topk_strategy=pop_topk_strategy,
            verbose=verbose,
        )
        # Suff to track for early stopping
        self._early_stopping_steps = early_stopping_steps
        self._last_population = None
        self._steps_since_new_population = 0
        ####################################################################
        self.prefix_ids = None
        if len(preprefix_str):
            self.preprefix_ids = torch.tensor(
                self.tokenizer.encode(preprefix_str, add_special_tokens=False), dtype=int
            ).to(device)
        else:
            self.preprefix_ids = torch.tensor([], dtype=int).to(device)
        
        prompt_str = preprefix_str.lstrip()
        prompt_str = (' ' + prompt_str) if len(prompt_str) else ''
        self._pre_data_token_ids = self._pre_data_token_ids = self.tokenizer(
           f"{llm_candidate_regeneration_prompt_start}\n\n", return_tensors='pt').input_ids.to(device)
        self._post_data_token_ids = self.tokenizer(
           f"\n\n{llm_candidate_regeneration_prompt_end}" + prompt_str, return_tensors='pt').input_ids.to(device)
        
        self.llm_candidate_regeneration_prompt_start = llm_candidate_regeneration_prompt_start
        self.llm_candidate_regeneration_prompt_end = llm_candidate_regeneration_prompt_end
        ####################################################################
        self._iprompt_verbose = verbose
        self._step = 0
        
    
    def serialize(self, eval_dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> Dict[str, Any]:
        r = super().serialize(eval_dataloader=eval_dataloader, possible_answer_mask=possible_answer_mask)
        r["topk_pop_sample"] = self._topk_pop_sample
        r["pop_size"] = self._pop_size
        r["num_mutations_per_ex"] = self._num_mutations_per_ex
        r["num_random_generations"] = self._num_random_generations
        r["generation_temp"] = self._generation_temp
        r["generation_top_p"] = self._generation_top_p
        r["generation_repetition_penalty"] = self._generation_repetition_penalty
        r["generation_bad_words_ids"] = self._generation_bad_words_ids
        r["pre_data_prompt_str"] = self.tokenizer.decode(self._pre_data_token_ids.flatten())
        r["post_data_prompt_str"] = self.tokenizer.decode(self._post_data_token_ids.flatten())
        return r
    
    def _initialize_pop_once(self, full_text_ids: torch.Tensor):
        if self._pop_initialized: return

        while len(self._prefix_pool) < self._pop_size:
            conditional_input_ids = random.choice(full_text_ids)[None]
            num_conditional_tokens = conditional_input_ids.numel()
            input_ids = self._generate(
                input_ids=conditional_input_ids,
                num_conditional_tokens=num_conditional_tokens
            ).squeeze()
            assert input_ids.numel() == self._num_tokens
            self._prefix_pool.initialize_prefix(input_ids)

        self._pop_initialized = True
    
    @property
    def _generation_model(self) -> transformers.AutoModelForCausalLM:
        """Returns the model to use for generation.

        We optionally support using different models for generation and discrimination.
        However, by default, we use the same model for both.
        """
        if self.other_generation_model:
            return self.other_generation_model
        else:
            return self.model
    
    def _generate(self, input_ids: torch.Tensor, num_conditional_tokens: int) -> torch.Tensor:
        """Generates some text using the model and preset hparams.

        If `num_conditional_tokens` > 0, generates extra text because there was an additional
        prefix set.
        """
        attention_mask = ~(input_ids == self.tokenizer.pad_token_id)
        assert attention_mask.shape == input_ids.shape

        if self._is_t5:
            output_length = self._num_tokens + 1 # will add pad token
        else:
            output_length = self._num_tokens + num_conditional_tokens
        
        # print("iPrompt._generate", input_ids.shape, "//", self.tokenizer.decode(input_ids[0]))
        
        g = self._generation_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_length=output_length,
            max_length=output_length,
            temperature=self._generation_temp,
            top_p=self._generation_top_p,
            repetition_penalty=self._generation_repetition_penalty,
            bad_words_ids=self._generation_bad_words_ids,
            do_sample=True
        )
        
        if self._is_t5:
            assert (g[:, 0] == 0).all()
            g = g[:, 1:]
        else:
            # Split off the conditional part, we only want the prefix part, which
            # starts after the conditional part.
            g = g[:, num_conditional_tokens:]

        if self._iprompt_verbose:
            # Print a random one (but remove padded tokens and newlines)
            idx = random.choice(range(len(input_ids)))
            # idx_attention_mask = torch.cat(
            #     (attention_mask[idx], torch.ones(self._num_tokens).to(device)), dim=0
            # ).bool()
            random_sentence_ids = g[idx]
            # print(">>", self.tokenizer.decode(random_sentence_ids).replace('\n', '\\n'))
        
        return g
    
    def _select_pop_topk(self, k: int, min_occurrences: int = None) -> List[Tuple[int]]:
        return self._prefix_pool.topk(k=k, min_occurrences=min_occurrences)

    def _track_early_stopping(self):
        """Track changes in population to tell when to stop early."""
        __n_early_stop = 5
        population = set(self._select_pop_topk(k=__n_early_stop, min_occurrences=3))
        if (len(population) == __n_early_stop) and (self._last_population == population):
            self._steps_since_new_population += 1
            if self._iprompt_verbose:
                print("self._steps_since_new_population:", self._steps_since_new_population)
        else:
            self._last_population = population
            self._steps_since_new_population = 0
            if self._iprompt_verbose:
                print("new population:", [self.tokenizer.decode(p) for p in sorted(population)])

    def check_early_stop(self) -> bool:
        """Allow prefix models to stop early."""
        if self._early_stopping_steps == -1:
            return False
        return self._steps_since_new_population >= self._early_stopping_steps
    
    def _get_population_and_random_generations(self, full_text_ids: torch.Tensor) -> torch.Tensor:
        population_pool = self._select_pop_topk(k=self._topk_pop_sample)
        # if self._iprompt_verbose:
            # print("population_pool:", [self.tokenizer.decode(p) for p in population_pool])
        population = random.sample(population_pool, self._pop_size)
        population = torch.tensor(population).to(device)

        if self._num_random_generations > 0:
            random_idxs = torch.randint(
                low=0, high=len(full_text_ids), size=(self._num_random_generations,)
            )
            random_full_text_ids = full_text_ids[random_idxs]
            num_conditional_tokens = full_text_ids.shape[1]
            random_population = self._generate(
                input_ids=random_full_text_ids,
                num_conditional_tokens=num_conditional_tokens
            )

            full_population = torch.cat((population, random_population), dim=0)
        else:
            # Support case where _num_random_generations is set to 0.
            full_population = population

        assert full_population.shape == (
            self._pop_size + self._num_random_generations,
            self._num_tokens
        )
        return full_population
    
    def _mutate(self, population_input_ids: torch.Tensor, full_text_ids: torch.Tensor) -> List[torch.Tensor]:
        """Mutates a population of prefixes.

        Truncates to a random place and then generates new options
        to try.

        Args:
            population_input_ids (int torch.Tensor): input IDs for each prefix in population
            full_text_ids (int torch.Tensor): input IDs for each data item in the batch. Intended
                be used to do prefix generation conditioned on data
        """
        assert population_input_ids.shape[1] == self._num_tokens
        input_ids = population_input_ids.repeat((self._num_mutations_per_ex, 1))

        self._roll_before_truncation = False
        if self._roll_before_truncation:
            roll_amount = random.randint(0, self._num_tokens-1)
            input_ids = torch.roll(input_ids, roll_amount, dims=[1])

        truncate_position = random.randint(0, self._num_tokens-1)
        truncated_input_ids = input_ids[:, :truncate_position]

        random_idxs = torch.randint(low=0, high=len(full_text_ids), size=(len(input_ids), ))
        random_full_text_ids = full_text_ids[random_idxs]
        conditional_input_ids = torch.cat((random_full_text_ids, truncated_input_ids), dim=1)

        num_conditional_tokens = full_text_ids.shape[1]
        new_input_ids = self._generate(
            input_ids=conditional_input_ids,
            num_conditional_tokens=num_conditional_tokens
        )
        return new_input_ids
    
    def _score_population(
            self, 
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            population_input_ids: torch.Tensor,
            possible_answer_mask: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scores a population of prefixes and updates `self._genetic_pool`."""
        pop_size = len(population_input_ids)
        all_candidate_losses = torch.zeros(pop_size, dtype=float).to(device)
        all_accuracy = torch.zeros(pop_size, dtype=float).to(device)
        all_candidate_n_correct = torch.zeros(pop_size, dtype=int).to(device)
        for i in range(pop_size):
            with torch.no_grad():
                _cand_input_ids, cand_loss, cand_n_correct = (
                    self._compute_loss_with_set_prefix(
                        original_input_ids=x_tokenized.input_ids,
                        next_token_ids=y_tokenized.input_ids,
                        possible_answer_mask=possible_answer_mask,
                        prefix_ids=population_input_ids[i],
                    )
                )
                cand_accuracy = cand_n_correct / len(x_tokenized.input_ids)
            all_candidate_n_correct[i] += cand_n_correct
            all_candidate_losses[i] += cand_loss
            all_accuracy[i] += cand_accuracy
        
        for i in range(pop_size):
            new_pop_input_ids = tuple(population_input_ids[i].cpu().tolist())
            assert len(new_pop_input_ids) == (self._num_tokens)
            self._prefix_pool.update(
                population_input_ids[i], all_candidate_losses[i], all_accuracy[i]
            )
        return all_candidate_losses, all_candidate_n_correct
    
    def _create_full_text_ids(
        self, full_text_input_ids: torch.Tensor) -> torch.Tensor:
        """Creates input for generating explanation.

        Takes tokenized inputs (like: "Input: 7 8 Output: 15")
        and makes a full string that looks like "Data:\n\n Input: .... 15 \n\nExplanation:\n\n",
        using whatever template is defined by pre-data and post-data.
        """
        B = len(full_text_input_ids)
        pre_data = self._pre_data_token_ids.repeat((B, 1)).to(device)  # Like "Data:\n\n"
        post_data = self._post_data_token_ids.repeat((B, 1)).to(device) # Like "\n\nPrompt:"
        output = torch.cat((pre_data, full_text_input_ids, post_data), dim=1)
        return output

    def compute_loss_and_call_backward(
            self,
            x_tokenized: transformers.BatchEncoding,
            y_tokenized: transformers.BatchEncoding,
            possible_answer_mask: torch.Tensor,
            full_text_tokenized: Optional[transformers.BatchEncoding] = None
        ) -> Tuple[torch.Tensor, int]:
        """Returns the loss from the best example in the population

        Note: does not call loss.backward()
        
        Returns:
            loss (float torch.Tensor) -- the loss
            num_correct (int): number of examples where prediction was correct
        """
        self.model.eval()

        # allow for conditioning only on x or y. This is mainly just used for ablations.
        if self.conditioning_strategy == "x_only":
            full_text_tokenized = y_tokenized
        elif self.conditioning_strategy == "y_only":
            full_text_tokenized = y_tokenized
        elif self.conditioning_strategy == "unconditional":
            full_text_tokenized['input_ids'] = torch.full(
                size=(len(y_tokenized), 1),
                fill_value=self.tokenizer.bos_token_id,
                device=device,
            )
            full_text_tokenized['attention_mask'] = torch.ones_like(
                full_text_tokenized['input_ids']
            )

        # logic here is that we want to see a sample multiple times before
        # we actually have a good estimate of its loss.
        num_min_occurrences = 2

        full_text_ids = self._create_full_text_ids(
            full_text_input_ids=full_text_tokenized.input_ids,
        )
        self._initialize_pop_once(full_text_ids=full_text_ids)

        prefix_save_folder = os.path.join(self.args.save_dir_unique, 'prefix')
        df_to_print = self._prefix_pool.print(topk=10, min_occurrences=num_min_occurrences)
        os.makedirs(prefix_save_folder, exist_ok=True)

        log_prefixes = False
        if log_prefixes:
            prefix_out_file = os.path.join(prefix_save_folder, f'prefix_{self._step}.p')
            df_to_print.to_pickle(prefix_out_file)
            print(f'wrote {len(df_to_print)} prefixes to {prefix_out_file}')

        # Grab new population
        population_input_ids = self._get_population_and_random_generations(
            full_text_ids=full_text_ids,
        )

        if self._num_mutations_per_ex > 0:
            mutated_population_input_ids = self._mutate(
                population_input_ids=population_input_ids, full_text_ids=full_text_ids
            )
            full_population_input_ids = torch.cat(
                (population_input_ids, mutated_population_input_ids), dim=0
            )
        else:
            # Support skipping mutation step by stetting _num_mutations_per_ex to 0
            full_population_input_ids = population_input_ids

        # Re-score new guys
        all_candidate_losses, all_candidate_n_correct = self._score_population(
            x_tokenized=x_tokenized,
            y_tokenized=y_tokenized,
            population_input_ids=full_population_input_ids,
            possible_answer_mask=possible_answer_mask
        )

        # Track changes in population to enable early stopping.
        self._track_early_stopping()

        # Reset prefix IDs so that the model can be readily used for eval.
        best_prefix_ids = min(self._prefix_pool._avg_loss, key=self._prefix_pool._avg_loss.get)
        self._set_prefix_ids(torch.tensor(best_prefix_ids).to(device))
        self.prefix_embedding.requires_grad = False

        self._step += 1
        return all_candidate_losses.min(), all_candidate_n_correct.max()
        
    def post_epoch(self, dataloader: torch.utils.data.DataLoader, possible_answer_mask: torch.Tensor) -> None:
        # 
        # Get candidate IDs for every position.
        # 
        pass
