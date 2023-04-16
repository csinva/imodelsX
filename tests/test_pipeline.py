from imodelsx import AugGAMClassifier
import datasets
import numpy as np

if __name__ == '__main__':
    # set up data
    dset = datasets.load_dataset('rotten_tomatoes')['train']
    dset = dset.select(np.random.choice(len(dset), size=10, replace=False))
    dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
    dset_val = dset_val.select(np.random.choice(
        len(dset_val), size=10, replace=False))

    # fit model
    m = AugGAMClassifier(
        checkpoint='textattack/distilbert-base-uncased-rotten-tomatoes',
        ngrams=2,
        all_ngrams=True,  # also use lower-order ngrams
        min_frequency=1
    )
    m.fit(dset['text'], dset['label'], batch_size=8)

    # predict
    preds = m.predict(dset_val['text'])
    print('acc_val', np.mean(preds == dset_val['label']))
    coefs_orig = np.array(list(m.coefs_dict_.values()))

    # check results when varying batch size
    m.fit(dset['text'], dset['label'], batch_size=16)
    preds_check = m.predict(dset_val['text'])
    assert np.allclose(preds, preds_check), 'predictions should be same when varying batch size'
    assert np.allclose(np.array(list(m.coefs_dict_.values())), coefs_orig), 'coefs should be same when varying batch size'

    # interpret
    print('Total ngram coefficients: ', len(m.coefs_dict_))
    print('Most positive ngrams')
    for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1], reverse=True)[:8]:
        print('\t', k, round(v, 2))
    print('Most negative ngrams')
    for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1])[:8]:
        print('\t', k, round(v, 2))

    print('successfully ran!')
