"""
.. include:: ../readme.md
"""

from .embgam.embgam import EmbGAMClassifier, EmbGAMRegressor
from .d3.d3 import explain_datasets_d3
from .iprompt.api import explain_dataset_iprompt
from .iprompt.data import get_add_two_numbers_dataset