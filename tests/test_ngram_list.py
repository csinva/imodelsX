from spacy.lang.en import English
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from imodelsx.util import generate_ngrams_list


def test_ngram_list():
    nlp = English()

    def spacy_tokenizer(doc):
        return [str(x) for x in nlp(doc)]
    for x in [
        'here is a sample movie review: it was a great movie, unbelievable that the cast was not nominated for oscars',
        'another movie review---did not enjoy it at all, the plot was boring and the acting was not good',
    ]:
        seqs = generate_ngrams_list(
            x, ngrams=2, tokenizer_ngrams=nlp, all_ngrams=True)
        seqs = np.sort(list(set(seqs)))

        # get feature_names when using sklearn countvectorizer on the sequence
        v = CountVectorizer(
            ngram_range=(1, 2), tokenizer=spacy_tokenizer, lowercase=True,
            token_pattern=None,
        )
        v.fit([x, x])
        seqs_out = v.get_feature_names_out().tolist()
        seqs_out = np.sort(list(set(seqs_out)))

        # print('seqs', seqs)
        # print('seqs_out', seqs_out)
        print('diff', [x for x in seqs if not x in seqs_out])
        print('diff2', [x for x in seqs_out if not x in seqs])
        print('sizes', seqs.size, seqs_out.size)

        assert np.all(np.sort(seqs) == np.sort(seqs_out)
                    ), 'generate_ngrams_list performs the same as countvectorizer'

if __name__ == '__main__':
    test_ngram_list()