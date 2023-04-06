"""
.. include:: ../readme.md
"""

from .auggam.auggam import AugGAMClassifier, AugGAMRegressor
from .augtree.tree import AugTreeClassifier, AugTreeRegressor
from .linear_finetune import LinearFinetuneClassifier, LinearFinetuneRegressor
from .d3.d3 import explain_datasets_d3
from .iprompt.api import explain_dataset_iprompt
from .iprompt.data import get_add_two_numbers_dataset