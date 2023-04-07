from typing import List
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import imodels
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import imodelsx.augtree.data
import imodelsx.augtree.llm
import imodelsx.augtree.augtree
import imodelsx.augtree.utils
from imodelsx.augtree.embed import EmbsManager
import imodelsx.util
from imodelsx.metrics import gini_score, gini_binary
import logging

STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'nor', 'only', 'own', 'so', 'than', 'too', 'very',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}


class Stump():
    def __init__(
        self,
        max_features=5,
        split_strategy='cart',
        tokenizer=None,
        refinement_strategy='None',
        use_refine_ties: bool=False,
        assert_checks: bool=False,
        llm_prompt_context: str='',
        embs_manager: EmbsManager=None,
        verbose: bool=True,
        use_stemming: bool=False,
        cache_expansions_dir: str=None,
    ):
        """Fit a single stump.
        Currently only supports binary classification with binary features.
        """
        self.max_features = max_features
        self.split_strategy = split_strategy
        self.use_refine_ties = use_refine_ties
        self.child_left = None
        self.child_right = None
        self.assert_checks = assert_checks
        self.llm_prompt_context = llm_prompt_context
        self.refinement_strategy = refinement_strategy
        self.verbose = verbose
        self.embs_manager = embs_manager
        self.use_stemming = use_stemming
        self.cache_expansions_dir = cache_expansions_dir
        if tokenizer is None:
            self.tokenizer = imodelsx.augtree.utils.get_spacy_tokenizer(use_stemming=use_stemming)
        else:
            self.tokenizer = tokenizer
        if self.split_strategy == 'cart':
            self.criterion = 'gini'
        elif self.split_strategy == 'id3':
            self.criterion = 'entropy'
        elif self.split_strategy == 'mse':
            self.criterion = 'mse'
            


    def fit(self, X, y, feature_names=None, X_text=None):
        # check input and set some attributes
        assert len(np.unique(y)) > 1, 'y should have more than 1 unique value'
        if not isinstance(self, RegressorMixin):
            assert len(np.unique(y)) <= 2, 'only binary classification is supported'
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # fit stump
        if self.split_strategy == 'linear':
            if isinstance(self, RegressorMixin):
                raise NotImplementedError('linear split strategy not implemented for regression')
            self.stump_keywords_idxs  = self._get_stump_keywords_linear(X, y)
        else:
            self.stump_keywords_idxs = self._get_stump_keywords_cart(X, y)
        self.stump_keywords = self.feature_names[self.stump_keywords_idxs]

        # set value
        self._set_value_acc_samples(X_text, y)
        if self.failed_to_split:
            return self

        # checks
        if self.assert_checks:
            preds_text = self.predict(X_text=X_text, predict_strategy='text')
            preds_tab = self.predict(X=X, predict_strategy='tabular')
            assert np.all(
                preds_text == preds_tab), 'predicting with text and tabular should give same results'
            assert self.value[1] > self.value[0], 'right child should have greater val than left but value=' + \
                str(self.value)
            assert self.value[1] > self.value_mean, 'right child should have greater val than parent ' + \
                str(self.value)

        # refine llm keywords
        if not self.refinement_strategy == 'None':
            if self.verbose:
                logging.debug(f'\t\tbefore refining acc {self.acc:0.4f}')
            self.stump_keywords_refined = self._refine_keywords(
                self.stump_keywords, X_text, y, tokenizer=self.tokenizer,
                )
            self._set_value_acc_samples(X_text, y)
            if self.verbose:
                logging.debug(f'\t\trefined acc {self.acc:.4f} {self.stump_keywords_refined[0]} -> {self.stump_keywords_refined[:5]}...')
        
        return self


    def predict(self, X=None, X_text: List[str] = None,
                predict_strategy='text', keywords=None) -> np.ndarray[int]:
        """Returns prediction 1 for positive and 0 for negative.
        """

        assert not (predict_strategy == 'tabular' and X is None)
        assert not (predict_strategy == 'text' and X_text is None)

        if predict_strategy == 'tabular':
            X = imodels.util.arguments.check_fit_X(X)
            # predict whether input has any of the features in stump_keywords_idxs
            X_feats = X[:, self.stump_keywords_idxs]
            pred = np.any(X_feats, axis=1)
            if self.pos_or_neg == 'pos':
                return pred.astype(int)
            else:
                return 1 - pred
        elif predict_strategy == 'text':
            if not keywords:
                if hasattr(self, 'stump_keywords_refined'):
                    keywords = self.stump_keywords_refined
                else:
                    keywords = self.stump_keywords
            ngrams_used_to_predict = max(
                    [len(keyword.split(' ')) for keyword in keywords])

            def contains_any_of_keywords(text):
                text = text.lower()
                text = imodelsx.util.generate_ngrams_list(
                    text,
                    ngrams=ngrams_used_to_predict,
                    tokenizer_ngrams=self.tokenizer,
                    all_ngrams=True
                )
                for keyword in keywords:
                    if keyword in text:
                        return 1
                return 0
            contains_keywords = 1 * \
                np.array([contains_any_of_keywords(x) for x in X_text])
            if self.pos_or_neg == 'pos':
                return contains_keywords
            else:
                return 1 - contains_keywords

    def predict_regression(self, X_text, **kwargs):
        preds_binary = self.predict(X_text=X_text, **kwargs)
        return preds_binary * self.value[1] + (1 - preds_binary) * self.value[0]

    def _get_stump_keywords_cart(self, X, y):
        '''iteratively select the feature selected by DecisionTreeClassifier
        removes that feature, and repeats
        '''

        if self.criterion == 'gini':
            criterion_func = imodelsx.augtree.utils.impurity_gini
        elif self.criterion == 'entropy':
            criterion_func = imodelsx.augtree.utils.impurity_entropy
        elif self.criterion == 'mse':
            criterion_func = imodelsx.augtree.utils.impurity_mse
        

        # Calculate the gini impurity reduction for each (binary) feature in X
        impurity_reductions = []

        # whether the feature increases the likelihood of the positive class
        feature_positive = []
        n = y.size
        gini_impurity = criterion_func(y)
        # assert gini_impurity_1 == gini_impurity, 'gini impurity should be the same'
        for i in range(X.shape[1]):
            x = X[:, i]
            idxs_r = x > 0.5
            idxs_l = x <= 0.5
            if idxs_r.sum() == 0 or idxs_l.sum() == 0:
                impurity_reductions.append(0)
                feature_positive.append(True)
            else:
                gini_impurity_l = criterion_func(y[idxs_l])
                gini_impurity_r = criterion_func(y[idxs_r])
                # print('l', indexes_l.sum(), 'r', indexes_r.sum(), 'n', n)
                impurity_reductions.append(
                    gini_impurity
                    - (idxs_l.sum() / n) * gini_impurity_l
                    - (idxs_r.sum() / n) * gini_impurity_r
                )
                feature_positive.append(np.mean(y[idxs_r]) > np.mean(y[idxs_l]))

        impurity_reductions = np.array(impurity_reductions)
        feature_positive = np.arange(X.shape[1])[np.array(feature_positive)]

        # find the top self.max_features with the largest impurity reductions
        args_largest_reduction_first = np.argsort(impurity_reductions)[::-1]
        self.impurity_reductions = impurity_reductions[args_largest_reduction_first][:self.max_features]
        # print('\ttop_impurity_reductions', impurity_reductions[args_largest_reduction_first][:5],
        #   'max', max(impurity_reductions))
        # print(f'\t{X.shape=}')
        imp_pos_top = [
            k for k in args_largest_reduction_first
            if k in feature_positive
            and not k in STOPWORDS
        ][:self.max_features]
        imp_neg_top = [
            k for k in args_largest_reduction_first
            if not k in feature_positive
            and not k in STOPWORDS
        ][:self.max_features]

        # feat = DecisionTreeClassifier(max_depth=1).fit(X, y).tree_.feature[0]
        if np.sum(imp_pos_top) > np.sum(imp_neg_top):
            self.pos_or_neg = 'pos'
            return imp_pos_top
        else:
            self.pos_or_neg = 'neg'
            return imp_neg_top

    def _refine_keywords(
        self,
        keywords: List[str], X_text: List[str],
        y, tokenizer,
        max_words_in_single_keyword=4,
    ) -> List[str]:
        """Refine each keyword using LLM.
        Greedily add extra keywords based on whether they improve acc (should change to impurity).
        Return list corresponding to the candidates from the single best keyword
        """
        
        if isinstance(self, RegressorMixin):
            criterion_func = imodelsx.augtree.utils.mse_score
            predict_func = self.predict_regression
        else:
            criterion_func = imodelsx.augtree.utils.gini_score
            predict_func = self.predict


        criterions_keyword = []
        candidates_list_keyword = []
        for i in range(len(keywords)):

            # get keyword
            keyword = keywords[i]

            # get refined_keywords
            if self.refinement_strategy == 'llm':
                keywords_refined = imodelsx.augtree.llm.expand_keyword(keyword, self.llm_prompt_context, cache_dir=self.cache_expansions_dir)
            elif self.refinement_strategy == 'embs':
                keywords_refined = self.embs_manager.expand_keyword(keyword)

            # filter out keywords that are too long
            keywords_refined = [
                k for k in keywords_refined
                if len(k.split()) <= max_words_in_single_keyword
            ]

            if self.use_stemming:
                # apply tokenizer to each unigram and combine
                keywords_refined = [
                    ' '.join([str(tok) for tok in tokenizer(str(word))])
                    for word in keywords_refined
                ]

                # drop things that end up too short
                keywords_refined = [
                    k for k in keywords_refined if len(k) > 2
                ]

            # greedily grow words one at a time, testing acc with each new word
            words = [keyword]
            criterion_max = criterion_func(y, predict_func(X_text=X_text, keywords=words))
            for keyword_refined in keywords_refined:
                preds_check = predict_func(
                    X_text=X_text, keywords=words + [keyword_refined])

                # if acc improved, add the refined_keyword
                crit = criterion_func(y, preds_check)
                if self.use_refine_ties:
                    check_crit = crit >= criterion_max
                else:
                    check_crit = crit > criterion_max
                if check_crit:
                    words.append(keyword_refined)
                    criterion_max = crit
                    logging.debug(f'\t\t\tadded {repr(keyword_refined)}')

            # append the results
            criterions_keyword.append(criterion_max)
            candidates_list_keyword.append(words)
            # print(f'\t\t {i} refined acc {acc_max:.4f}',
                #   keyword, '->', words[:5], '...')

        idx_best = np.argmax(criterions_keyword)
        keywords_best = [
            ' '.join([str(tok) for tok in tokenizer(str(word))])  # clean up word
            for word in candidates_list_keyword[idx_best]
        ]
        # print(f'\trefined acc {criterion_max:.4f}', keywords[idx_best], '->', keywords_best[:5], '...') 

        return keywords_best

    def _set_value_acc_samples(self, X_text, y):
        """Set value and accuracy of stump.
        """
        idxs_right = self.predict(X_text=X_text).astype(bool)
        n_right = idxs_right.sum()
        if n_right == 0 or n_right == y.size:
            self.failed_to_split = True
            return
        else:
            self.failed_to_split = False
        self.n_samples = [y.size - n_right, n_right]
        self.value = [np.mean(y[~idxs_right]), np.mean(y[idxs_right])]
        self.value_mean = np.mean(y)
        if isinstance(self, RegressorMixin):
            preds = self.value[1] * idxs_right + self.value[0] * ~idxs_right
            self.acc = imodelsx.augtree.utils.mse_score(y, preds)
        else:
            preds = 1 * idxs_right
            self.acc = accuracy_score(y, preds)
        # self.impurity_reduction = gini_binary(y.mean()) - \
            # gini_binary(self.value[1]) / self.n_samples[1] * y.size - \
            # gini_binary(self.value[0]) / self.n_samples[0] * y.size

    def __str__(self):
        if hasattr(self, 'stump_keywords_refined'):
            keywords = self.stump_keywords_refined
        else:
            keywords = self.stump_keywords
        keywords_str = ", ".join(keywords[:5])
        if len(keywords) > 5:
            keywords_str += f'...({len(keywords) - 5} more)'
        sign = {'pos': '+', 'neg': '--'}[self.pos_or_neg]
        return f'Stump(val={self.value_mean:0.2f} n={self.n_samples}) {sign} {keywords_str}'

    def get_str_simple(self):
        if hasattr(self, 'stump_keywords_refined'):
            keywords = self.stump_keywords_refined
        else:
            keywords = self.stump_keywords
        keywords_str = ", ".join(keywords[:5])
        if len(keywords) > 5:
            keywords_str += f'...({len(keywords) - 5} more)'
        sign = {'pos': '+', 'neg': '--'}[self.pos_or_neg]
        return f'{sign} {keywords_str}'
    
    def _get_stump_keywords_linear(self, X, y):
        # fit a linear model
        m = LogisticRegression().fit(X, y)
        m.fit(X, y)

        # find the largest magnitude coefs
        abs_feature_idxs = m.coef_.argsort().flatten()
        bot_feature_idxs = abs_feature_idxs[:self.max_features]
        top_feature_idxs = abs_feature_idxs[-self.max_features:][::-1]

        # return the features with the largest magnitude coefs
        if np.sum(abs(bot_feature_idxs)) > np.sum(abs(top_feature_idxs)):
            self.pos_or_neg = 'neg'
            return bot_feature_idxs
        else:
            self.pos_or_neg = 'pos'
            return top_feature_idxs

class StumpRegressor(Stump, RegressorMixin):
    ...


class StumpClassifier(Stump, ClassifierMixin):
    ...
