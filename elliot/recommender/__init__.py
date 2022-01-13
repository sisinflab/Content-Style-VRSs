"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel

from .latent_factor_models import BPRMF_batch
from .visual_recommenders import VBPR, DeepStyle, ACF, DVBPR, VNPR
from .neural import NeuMF
from .proposed import CSV

