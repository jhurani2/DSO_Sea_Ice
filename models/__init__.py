"""DSO_Sea_Ice models package.

Expose dataset, model, training & utility helpers.
"""

from .data import XarrayImageDataset, get_dataloaders
from .models import PersistenceModel, LinearBaseline, SimpleCNN, ResNetEncoderDecoder
from .train import Trainer
from .utils import mae, rmse, r2_score, save_checkpoint, load_checkpoint

__all__ = [
    'XarrayImageDataset', 'get_dataloaders',
    'PersistenceModel', 'LinearBaseline', 'SimpleCNN', 'ResNetEncoderDecoder',
    'Trainer',
    'mae', 'rmse', 'r2_score', 'save_checkpoint', 'load_checkpoint'
]
