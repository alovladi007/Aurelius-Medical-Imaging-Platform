"""Training modules for cancer histopathology models."""

from cancer_quant_model.training.train_loop import Trainer
from cancer_quant_model.training.eval_loop import Evaluator

__all__ = ["Trainer", "Evaluator"]
