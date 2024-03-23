import imodelsx

if __name__ == "__main__":
    checkpoints = [
        'gpt2', 'meta-llama/Llama-2-7b-hf',
        'mistralai/Mistral-7B-v0.1', 'EleutherAI/gpt-j-6b']
    prompts = ['may the force be with', 'roses are red, violets are']
    answers = [' you', ' blue']
    for checkpoint in checkpoints:
        for i in range(len(prompts)):
            llm = imodelsx.llm.get_llm(checkpoint)
            output = llm(prompts[i])
            assert output.startswith(
                answers[i]), f'output for {checkpoint} should start with "{answers[i]}" but instead got "{output}"'
        outputs = llm(prompts)
        assert len(outputs) == len(prompts), 'length mismatch'
        for i in range(len(prompts)):
            assert outputs[i].startswith(
                answers[i]), f'output for {checkpoint} should start with "{answers[i]}" but instead got "{outputs[i]}"'
    print('success for', checkpoints)
