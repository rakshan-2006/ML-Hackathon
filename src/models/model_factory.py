"""Model factory: each model lives in src/models/<name>.py and exposes

    NAME, FAMILY, DEFAULT_PARAMS, SCALE_FEATURES, CLASS_IMBALANCE
    build_estimator(params, seed) -> sklearn classifier with predict_proba

Modules are found by name, so adding a model never edits this file.
"""

import importlib
import re

from src.training.preprocessing import make_pipeline


def model_module(name):
    if not re.fullmatch(r"[a-z_]+", name):
        raise ValueError(f"invalid model name: {name!r}")
    try:
        return importlib.import_module(f"src.models.{name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"unknown model: {name!r}") from exc


def model_params(name, overrides=None):
    module = model_module(name)
    unknown = set(overrides or {}) - set(module.DEFAULT_PARAMS)
    if unknown:
        raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
    return {**module.DEFAULT_PARAMS, **(overrides or {})}


def get_model(name, params=None, seed=2026):
    """Full Pipeline (preprocessing + estimator) for one model."""
    module = model_module(name)
    estimator = module.build_estimator(model_params(name, params), seed)
    return make_pipeline(estimator, scale=module.SCALE_FEATURES)
