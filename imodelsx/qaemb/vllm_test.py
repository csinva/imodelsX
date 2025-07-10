import imodelsx.llm
import imodelsx.util
import os
from neuro.features.questions.gpt4 import QS_35_STABLE
import time


def time_batch_inference_vllm(prompts):
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    ray.init(log_to_driver=False)
    ray.data.DataContext.get_current().enable_progress_bars = False
    ds = ray.data.from_items(
        [{"text": prompt} for prompt in prompts],
        # parallelism=1,
        # ray_remote_args={"num_cpus": 1, "num_gpus": 0},
    )

    t0 = time.time()
    config = vLLMEngineProcessorConfig(
        model_source='meta-llama/Meta-Llama-3-8B-Instruct',
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 8192,
        },
        concurrency=1,
        batch_size=64,
        tensor_parallel_size=4,
    )
    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": row["text"]},
            ],
            sampling_params=dict(
                temperature=0,
                max_tokens=1,
            ),
        ),
        postprocess=lambda row: dict(
            answer=row["generated_text"],
            **row,  # This will return all the original columns in the dataset.
        ),
    )
    t1 = time.time()
    print(f'vLLM processor created in {t1 - t0:.2f} seconds')
    
    ds = vllm_processor(ds)
    outputs = ds.take(limit=10)
    t2 = time.time()
    print(f'Inference completed in {t2 - t1:.2f} seconds')
    ray.shutdown()

def time_batch_inference_hf(prompts):
    # set enable_prefix_caching=True to enable APC
    t0 = time.time()
    llm = imodelsx.llm.get_llm(
        checkpoint='meta-llama/Meta-Llama-3-8B-Instruct',
    )
    t1 = time.time()
    print(f'LLM created in {t1 - t0:.2f} seconds')
    outputs = llm(prompts)
    t2 = time.time()
    print(f'Inference completed in {t2 - t1:.2f} seconds')
    

story_text = open(os.path.expanduser('~/automated-brain-explanations/data/example_story_with_punctuation.txt'), 'r').read()
ngrams = imodelsx.util.generate_ngrams_list(story_text, ngrams=10)

if __name__ == '__main__':
    prompt_template = 'Input: {example}\nQuestion: {question} Answer yes or no.'
    question = QS_35_STABLE[0]
    prompts = [prompt_template.format(example=ngram, question=question) for ngram in ngrams]

    prompt_template2 = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nInput: {example}\nQuestion: {question}\nAnswer yes or no.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    prompts2 = [prompt_template2.format(example=ngram, question=question) for ngram in ngrams]

    time_batch_inference_vllm(prompts)
    # time_batch_inference_hf(prompts2)
