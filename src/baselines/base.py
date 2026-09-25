"""Base class for all forecasting models in the comparison framework."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseForecastingModel(ABC):
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.model = None
        self.is_trained = False
        self.results: Dict = {}

    @abstractmethod
    def build(self):
        ...

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, run_logger=None):
        ...

    @abstractmethod
    def predict(self, X):
        ...

    @abstractmethod
    def evaluate(self, X_test, y_test, **kwargs):
        ...

    def save_model(self, path: str):
        # BUG (caught by tests/test_smoke.py's full CLI integration run,
        # not by the unit tests): torch.save() does not create missing
        # parent directories, and nothing upstream created models/ before
        # the first checkpoint write. Every training run that saved a
        # checkpoint on first improvement (i.e. every run) would crash at
        # that point unless the caller happened to mkdir first.
        import os
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        self.is_trained = True
        logger.info(f"Model loaded from {path}")
