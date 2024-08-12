"""
Trigger-based Language Model Project

This package contains modules for generating datasets, training models,
and evaluating performance for a trigger-based language model system.
"""

from .models import load_model, load_tokenizer, TriggerClassifier
from .data import generate_dataset
from .training import supervised_fine_tuning, hh_rlhf_training
from .utils import evaluation

__all__ = [
    'load_model', 'load_tokenizer', 'TriggerClassifier', 'generate_dataset',
    'supervised_fine_tuning', 'hh_rlhf_training',
    'evaluation', 'star_gate_config', 'conversation_pipeline'
]