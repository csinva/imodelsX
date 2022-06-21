# baselines

- [fasttext](https://adityaroc.medium.com/understanding-fasttext-an-embedding-to-look-forward-to-3ee9aa08787#:~:text=Fasttext%20can%20generate%20embedding%20for,representation%20in%20the%20training%20set.) - sums partial embeddings of a word to make up the whole word

# follow-up experiments

Probably later want to switch to "bert-base-uncased", ""distilbert-base-uncased-finetuned-sst-2-english" + try different layers + train custom dnn for this task.

- could include positional information...

- GAM version of this - train a different net for single-word embedding, multi-word, etc....