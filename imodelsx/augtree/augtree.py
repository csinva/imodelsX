from typing import Dict, List
import numpy as np
import imodels
import imodelsx.augtree.llm
from imodelsx.augtree.embed import EmbsManager
from imodelsx.augtree.stump import Stump, StumpClassifier, StumpRegressor
import logging
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class AugTree:
    def __init__(
        self,
        max_depth: int = 3,
        max_features=5,
        split_strategy='cart',
        refinement_strategy='None',
        verbose=True,
        tokenizer=None,
        use_refine_ties=False,
        assert_checks=False,
        llm_prompt_context: str='',
        use_stemming=False,
        embs_manager: EmbsManager=None,
        cache_expansions_dir: str=None,
    ):
        '''
        Params
        ------
        max_depth: int
            Maximum depth of the tree.
        max_features: int
            Number of features to consider expanding at each stump
        split_strategy: str
            Strategy for generating candidate seed keyphrases.
        refinement_strategy: str
            'None', 'llm', or 'embs'
        verbose: bool
            Whether to print debug statements
        tokenizer
            Tokenizer to use for splitting text into tokens
        use_refine_ties: bool
            Whether to include expanded keywords that don't improve or decrease performance
        assert_checks: bool
            Whether to run checks during fitting
        llm_prompt_context: str
            Extra context string provided llm_refine (if refinement_strategy=llm)
        embs_manager
            Class that provides function to query for keywords from closest embeddings
        cache_expansions_dir: str
            Directory to cache keyphrase expansions
        '''
        self.max_depth = max_depth
        self.max_features = max_features
        self.split_strategy = split_strategy
        self.verbose = verbose
        self.use_refine_ties = use_refine_ties
        self.assert_checks  = assert_checks
        self.llm_prompt_context = llm_prompt_context
        self.refinement_strategy = refinement_strategy
        self.use_stemming = use_stemming
        self.embs_manager = embs_manager
        self.cache_expansions_dir = cache_expansions_dir
        if tokenizer is None:
            self.tokenizer = imodelsx.augtree.utils.get_spacy_tokenizer(use_stemming=use_stemming)
        else:
            self.tokenizer = tokenizer
        assert self.refinement_strategy in ['None', 'llm', 'embs']
        if self.refinement_strategy == 'embs':
            assert embs_manager is not None, 'must pass embs_manager when refinement_strategy == "embs"'

    def fit(self, X=None, y=None, feature_names=None, X_text=None):
        if X is None and X_text:
            warnings.warn("X is not passed, defaulting to generating unigrams from X_text")
            X, _, feature_names = imodelsx.augtree.data.convert_text_data_to_counts_array(X_text, [], ngrams=1)

        # check and set some attributes
        X, y, _ = imodels.util.arguments.check_fit_arguments(
            self, X, y, feature_names)
        if isinstance(X_text, list):
            X_text = np.array(X_text).flatten()
        self.feature_names = feature_names
        if isinstance(self.feature_names, list):
            self.feature_names = np.array(self.feature_names).flatten()

        # fit root stump
        stump_kwargs = dict(
            split_strategy=self.split_strategy,
            max_features=self.max_features,
            tokenizer=self.tokenizer,
            use_refine_ties=self.use_refine_ties,
            assert_checks=self.assert_checks,
            llm_prompt_context=self.llm_prompt_context,
            refinement_strategy=self.refinement_strategy,
            embs_manager = self.embs_manager,
            verbose=self.verbose,
            use_stemming=self.use_stemming,
            cache_expansions_dir=self.cache_expansions_dir,
        )

        # assume that the initial split finds a feature that provides some benefit
        # otherwise, one leaf will end up NaN
        if isinstance(self, RegressorMixin):
            stump_class = StumpRegressor
        else:
            stump_class = StumpClassifier
        stump = stump_class(**stump_kwargs).fit(
            X, y,
            feature_names=self.feature_names,
            X_text=X_text
        )
        stump.idxs = np.ones(X.shape[0], dtype=bool)
        self.root_ = stump

        # recursively fit stumps and store as a decision tree
        stumps_queue = [stump]
        i = 0
        depth = 1
        while depth < self.max_depth:
            stumps_queue_new = []
            for stump in stumps_queue:
                stump = stump
                if self.verbose:
                    logging.debug(f'Splitting on depth={depth} stump_num={i} {stump.idxs.sum()}')
                idxs_pred = stump.predict(X_text=X_text) > 0.5
                for idxs_p, attr in zip([~idxs_pred, idxs_pred], ['child_left', 'child_right']):
                    # for idxs_p, attr in zip([idxs_pred], ['child_right']):
                    idxs_child = stump.idxs & idxs_p
                    if self.verbose:
                        logging.debug(f'\t{idxs_pred.sum()} {idxs_child.sum()}', len(np.unique(y[idxs_child])))
                    if idxs_child.sum() > 0 \
                        and idxs_child.sum() < stump.idxs.sum() \
                            and len(np.unique(y[idxs_child])) > 1:

                        # fit a potential child stump
                        stump_child = stump_class(**stump_kwargs).fit(
                            X[idxs_child], y[idxs_child],
                            X_text=X_text[idxs_child],
                            feature_names=self.feature_names,
                        )

                        # make sure the stump actually found a non-trivial split
                        if not stump_child.failed_to_split:
                            
                            # set the child stump
                            stump_child.idxs = idxs_child
                            acc_tree_baseline = np.mean(self.predict(
                                X_text[idxs_child]) == y[idxs_child])
                            if attr == 'child_left':
                                stump.child_left = stump_child
                            else:
                                stump.child_right = stump_child
                            stumps_queue_new.append(stump_child)
                            if self.verbose:
                                logging.debug(f'\t\t {stump.stump_keywords} {stump.pos_or_neg}')
                            i += 1

                        ######################### checks ###########################
                            if self.assert_checks and isinstance(self, ClassifierMixin):
                                # check acc for the points in this stump
                                acc_tree = np.mean(self.predict(
                                    X_text[idxs_child]) == y[idxs_child])
                                assert acc_tree >= acc_tree_baseline, f'stump acc {acc_tree:0.3f} should be > after adding child {acc_tree_baseline:0.3f}'

                                # check total acc
                                acc_total_baseline = max(y.mean(), 1 - y.mean())
                                acc_total = np.mean(self.predict(X_text) == y)
                                assert acc_total >= acc_total_baseline, f'total acc {acc_total:0.3f} should be > after adding child {acc_total_baseline:0.3f}'

                                # check that stumptrain acc improved over this set
                                # not necessarily going to improve total acc, since the stump always predicts 0/1
                                # even though the correct answer might be always 0 or always be 1
                                acc_child_baseline = min(
                                    y[idxs_child].mean(), 1 - y[idxs_child].mean())
                                assert stump_child.acc > acc_child_baseline, f'acc {stump_child.acc:0.3f} should be > baseline {acc_child_baseline:0.3f}'


            stumps_queue = stumps_queue_new
            depth += 1

        return self

    def predict_proba(self, X_text: List[str] = None):
        preds = []
        for x_t in X_text:

            # prediction for single point
            stump = self.root_
            while stump:
                # 0 or 1 class prediction here
                pred = stump.predict(X_text=[x_t])[0]
                value = stump.value

                if pred > 0.5:
                    stump = stump.child_right
                    value = value[1]
                else:
                    stump = stump.child_left
                    value = value[0]

                if stump is None:
                    preds.append(value)
        preds = np.array(preds)
        probs = np.vstack((1 - preds, preds)).transpose()  # probs (n, 2)
        return probs
    

    def predict(self, X_text: List[str] = None) -> np.ndarray[int]:
        preds_continuous = self.predict_proba(X_text)[:, 1]
        if isinstance(self, ClassifierMixin):
            return (preds_continuous > 0.5).astype(int)
        else:
            return preds_continuous

    def get_tree_dict_repr(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary representation of the tree
        Each key is a binary prefix string
            "0" for root
            "00" for left child of root
            "01" for right child of root
            "000" for left child of left child of root, etc.
        Each value is a list of strings, where each string is a keyword
        """
        tree_dict = {}
        stumps_queue = [(self.root_, "0")]
        while stumps_queue:
            stump, stump_id = stumps_queue.pop(0)
            
            # skip leaf nodes
            if stump.child_left is None and stump.child_right is None:
                continue
            
            if hasattr(stump, 'stump_keywords_refined'):
                keywords = stump.stump_keywords_refined
            else:
                keywords = stump.stump_keywords
            tree_dict[stump_id] = keywords
            if stump.child_left:
                stumps_queue.append((stump.child_left, stump_id + "0"))
            if stump.child_right:
                stumps_queue.append((stump.child_right, stump_id + "1"))
        return tree_dict

    def __str__(self):
        s = f'> Tree(max_depth={self.max_depth} max_features={self.max_features} refine={self.refinement_strategy})\n> ------------------------------------------------------\n'
        return s + self.viz_tree()

    def viz_tree(self, stump: Stump=None, depth: int=0, s: str='') -> str:
        if stump is None:
            stump = self.root_
        s += '   ' * depth + str(stump) + '\n'
        if stump.child_left:
            s += self.viz_tree(stump.child_left, depth + 1)
        else:
            s += '   ' * (depth + 1) + f'Neg n={stump.n_samples[0]} val={stump.value[0]:0.3f}' + '\n'
        if stump.child_right:
            s += self.viz_tree(stump.child_right, depth + 1)
        else:
            s += '   ' * (depth + 1) + f'Pos n={stump.n_samples[1]} val={stump.value[1]:0.3f}' + '\n'
        return s

class AugTreeRegressor(AugTree, RegressorMixin):
    ...


class AugTreeClassifier(AugTree, ClassifierMixin):
    ...