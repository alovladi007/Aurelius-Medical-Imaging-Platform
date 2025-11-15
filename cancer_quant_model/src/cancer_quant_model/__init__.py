"""
Cancer Quantitative Model Package

Production-ready quantitative cancer histopathology modeling pipeline.
"""

__version__ = "0.1.0"
__author__ = "Aurelius Medical Imaging Platform"
__license__ = "MIT"

from cancer_quant_model import config, data, models, training, explainability, api

__all__ = [
    "config",
    "data",
    "models",
    "training",
    "explainability",
    "api",
]
