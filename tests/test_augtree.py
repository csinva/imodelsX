import numpy as np
import imodelsx.augtree.data
import imodelsx.augtree.llm
import imodelsx.augtree.augtree
import random
import imodelsx.data
import imodelsx.augtree.ensemble


def seed_and_get_tiny_data(seed=1, classification_or_regression='classification'):
    np.random.seed(seed)
    random.seed(seed)
    if classification_or_regression == 'classification':
        X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
            dataset_name='rotten_tomatoes', subsample_frac=0.05, return_lists=True)
        X_train, _, feature_names = \
            imodelsx.augtree.data.convert_text_data_to_counts_array(
                X_train_text, X_test_text, ngrams=1)
    elif classification_or_regression == 'regression':
        X_train_text, X_test_text, y_train, y_test = imodelsx.data.load_huggingface_dataset(
            dataset_name='csinva/fmri_language_responses', subsample_frac=0.05, return_lists=True, label_name='voxel_0')
        X_train, _, feature_names = \
            imodelsx.augtree.data.convert_text_data_to_counts_array(
                X_train_text, X_test_text, ngrams=1)
    return X_train_text, X_test_text, y_train, X_train, y_test, feature_names


def test_stump_always_improves_acc():
    for classification_or_regression in ['classification', 'regression']: # classification, regression
        stump_class = imodelsx.augtree.stump.StumpClassifier if classification_or_regression == 'classification' else imodelsx.augtree.stump.StumpRegressor
        for seed in range(2):
            X_train_text, X_test_text, y_train, X_train, y_test, feature_names = \
                seed_and_get_tiny_data(seed=seed, classification_or_regression=classification_or_regression)
            m = stump_class().fit(
                X_train, y_train, feature_names, X_train_text)

            if classification_or_regression == 'classification':
                preds = m.predict(X_text=X_train_text)
                acc_baseline = max(y_train.mean(), 1 - y_train.mean())
                acc = np.mean(preds == y_train)
                assert acc > acc_baseline, 'stump must improve train acc'
                print(acc, acc_baseline)
            else:
                preds = m.predict_regression(X_text=X_train_text)
                # print('preds', preds)
                mse_baseline = np.mean((y_train - y_train.mean())**2)
                mse = np.mean((preds - y_train)**2)
                assert mse < mse_baseline, 'stump must improve train mse'
                print(mse, mse_baseline)

            # test prediction
            preds_text = m.predict(X_text=X_train_text, predict_strategy='text')
            preds_tab = m.predict(X=X_train, predict_strategy='tabular')
            assert np.all(
                preds_text == preds_tab), 'predicting with text and tabular should give same results'


def test_tree_monotonic_in_depth(refinement_strategy='None', max_features=1, embs_manager=None):
    for classification_or_regression in ['regression', 'classification']:
        X_train_text, X_test_text, y_train, X_train, y_test, feature_names = \
            seed_and_get_tiny_data(classification_or_regression=classification_or_regression)
        if classification_or_regression == 'classification':
            tree_class = imodelsx.augtree.augtree.AugTreeClassifier
        else:
            tree_class = imodelsx.augtree.augtree.AugTreeRegressor

        accs = [y_train.mean()]
        # for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        for max_depth in [1, 2, 3]:
            m = tree_class(
                max_depth=max_depth,
                split_strategy='cart',
                max_features=max_features,
                verbose=False,
                refinement_strategy=refinement_strategy,
                assert_checks=True,
                embs_manager=embs_manager,
            )
            m.fit(X_train, y_train, feature_names, X_train_text)
            preds = m.predict(X_text=X_train_text)
            accs.append(np.mean(preds == y_train))
            assert accs[-1] >= accs[-2], 'train_acc must monotonically increase with max_depth ' + \
                str(accs)
            print(m)
            print('\n')

def test_tree_ensemble(
    n_estimators=2,
    max_depth=2, max_features=1, refinement_strategy='None'):
    X_train_text, X_test_text, y_train, X_train, y_test, feature_names = seed_and_get_tiny_data()
    tree = imodelsx.augtree.augtree.AugTreeClassifier(
        max_depth=max_depth,
        split_strategy='cart',
        max_features=max_features,
        verbose=False,
        refinement_strategy=refinement_strategy,
        assert_checks=True,
    )
    m = imodelsx.augtree.ensemble.BaggingEstimatorText(tree, n_estimators=n_estimators)
    m.fit(X_train, y_train, X_text=X_train_text, feature_names=feature_names)
    acc = np.mean(m.predict(X_test_text) == y_test)
    assert acc >= 0.4, 'ensemble acc should not be too low'



if __name__ == '__main__':
    # test_stump_always_improves_acc()


    for refinement_strategy in ['None', 'llm']: #['None', 'llm']:
        test_tree_monotonic_in_depth(
            refinement_strategy=refinement_strategy,
            max_features=1)
    
    # embs_manager = imodelsx.augtree.embed.EmbsManager()
    # test_tree_monotonic_in_depth(
    #     refinement_strategy='embs',
    #     max_features=1,
    #     embs_manager=embs_manager
    #     )
    
    # test_tree_ensemble()