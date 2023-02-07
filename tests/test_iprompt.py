import torch

from imodelsx import explain_dataset_iprompt, get_add_two_numbers_dataset


if __name__ == '__main__':
    has_gpu = torch.cuda.device_count() > 0
    device = 'cuda' if has_gpu else 'cpu'
    llm_float16 = True if has_gpu else False


    # get a simple dataset of adding two numbers
    input_strings, output_strings = get_add_two_numbers_dataset(num_examples=100)
    for i in range(5):
        print(repr(input_strings[i]), repr(output_strings[i]))

    # explain the relationship between the inputs and outputs
    # with a natural-language prompt string
    prompts, metadata = explain_dataset_iprompt(
        input_strings=input_strings,
        output_strings=output_strings,
        checkpoint='EleutherAI/gpt-j-6B', # which language model to use
        num_learned_tokens=6, # how long of a prompt to learn
        n_shots=5, # number of examples in context
        n_epochs=15, # how many epochs to search
        verbose=1, # how much to print
        batch_size=16, # batch size for iprompt
        llm_float16=llm_float16, # whether to load the model in float_16
    )
