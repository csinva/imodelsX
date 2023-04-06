from typing import List
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import imodels
import imodelsx
import imodelsx.augtree.tree
import imodelsx.augtree.embed
import imodelsx.augtree.ensemble
from sklearn.ensemble import BaggingClassifier, BaggingRegressor


def get_model(args):
    if args.model_name == 'llm_tree':
        LLM_PROMPT_CONTEXTS = {
            'sst2': ' in the context of movie reviews',
            'rotten_tomatoes': ' in the context of movie reviews',
            'imdb': ' in the context of movie reviews',
            'financial_phrasebank': ' in the context of financial sentiment',
            'ag_news': ' in the context of news headlines',
            'tweet_eval': ' in the context of tweets',
            'emotion': ' in the context of tweet sentiment',
        }
        if args.use_llm_prompt_context:
            llm_prompt_context = LLM_PROMPT_CONTEXTS[args.dataset_name]
        else:
            llm_prompt_context = ''
        if args.refinement_strategy == 'embs':
            embs_manager = imodelsx.augtree.embed.EmbsManager(
                dataset_name=args.dataset_name,
                ngrams=args.ngrams,
                # metric=args.embs_refine_metric,
            )
        else:
            embs_manager = None
        if args.classification_or_regression == 'classification':
            cls = imodelsx.augtree.tree.TreeClassifier
        else:
            cls = imodelsx.augtree.tree.TreeRegressor
        model = cls(
            max_depth=args.max_depth,
            max_features=args.max_features,
            refinement_strategy=args.refinement_strategy,
            split_strategy=args.split_strategy,
            use_refine_ties=args.use_refine_ties,
            llm_prompt_context=llm_prompt_context,
            verbose=args.use_verbose,
            embs_manager=embs_manager,
            use_stemming=args.use_stemming,
        )
    elif args.model_name == 'decision_tree':
        if args.classification_or_regression == 'classification':
            model = DecisionTreeClassifier(max_depth=args.max_depth)
        else:
            model = DecisionTreeRegressor(max_depth=args.max_depth)
    elif args.model_name == 'c45':
        model = imodels.C45TreeClassifier(max_rules=int(2**args.max_depth) - 1)
    elif args.model_name == 'id3':
        model = DecisionTreeClassifier(max_depth=args.max_depth, criterion='entropy')
    elif args.model_name == 'hstree':
        estimator_ = DecisionTreeClassifier(max_depth=args.max_depth)
        model = imodels.HSTreeClassifier(estimator_=estimator_)
    elif args.model_name == 'ridge':
        model = RidgeClassifier(alpha=args.alpha)
    elif args.model_name == 'rule_fit':
        model = imodels.RuleFitClassifier(max_rules=args.max_rules)
    elif args.model_name == 'linear_finetune':
        if args.classification_or_regression == 'classification':
            model = imodelsx.LinearFinetuneClassifier()
        else:
            model = imodelsx.LinearFinetuneRegressor()
    else:
        raise ValueError(f'Invalid model_name: {args.model_name}')

    # make baggingclassifier
    if args.n_estimators > 1:
        if args.model_name == 'llm_tree':
            model = imodelsx.augtree.ensemble.BaggingEstimatorText(model, n_estimators=args.n_estimators)
        elif args.model_name == 'decision_tree':
            if args.classification_or_regression == 'classification':
                model = BaggingClassifier(model, n_estimators=args.n_estimators)
            else:
                model = BaggingRegressor(model, n_estimators=args.n_estimators)        

    return model
