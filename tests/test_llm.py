import imodelsx

if __name__ == "__main__":
    checkpoints = [
        'gpt2', 'meta-llama/Llama-2-7b-hf',
        'mistralai/Mistral-7B-v0.1', 'EleutherAI/gpt-j-6b']
    prompt = 'may the force be with'
    ans = ' you'
    for checkpoint in checkpoints:
        llm = imodelsx.llm.get_llm(checkpoint)
        output = llm(prompt)
        assert output.startswith(
            ans), f'output for {checkpoint} should start with "{ans}" but instead got "{output}"'
    print('success for', checkpoints)
