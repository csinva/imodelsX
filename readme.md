# baselines

- [fasttext](https://adityaroc.medium.com/understanding-fasttext-an-embedding-to-look-forward-to-3ee9aa08787#:~:text=Fasttext%20can%20generate%20embedding%20for,representation%20in%20the%20training%20set.) - sums partial embeddings of a word to make up the whole word
    - [paper](https://arxiv.org/pdf/1607.01759.pdf) - they do exactly this
- similar to [dirtycat](https://www.linkedin.com/feed/update/urn:li:activity:6944476456701820928?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A6944476456701820928%29)

# follow-up experiments

Probably later want to switch to "bert-base-uncased", ""distilbert-base-uncased-finetuned-sst-2-english" + try different layers + train custom dnn for this task.

- could include positional information...

- GAM version of this - train a different net for single-word embedding, multi-word, etc....