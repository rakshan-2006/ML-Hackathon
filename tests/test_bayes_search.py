import math

import optuna
import pandas as pd
import pytest
import yaml

from src.common.paths import PROJECT_ROOT
from src.models.model_factory import model_params
from src.optimization.bayes_search import suggest, trial_table

SPACE = yaml.safe_load((PROJECT_ROOT / "configs" / "optimisation.yaml").read_text())


def _study(n_startup=2):
    return optuna.create_study(sampler=optuna.samplers.TPESampler(
        seed=0, n_startup_trials=n_startup, multivariate=True, group=True))


def test_configured_space_produces_valid_gradient_boosting_params():
    space = SPACE["search_space"][SPACE["model"]]
    study = _study()
    for _ in range(6):
        trial = study.ask()
        params = model_params(SPACE["model"], suggest(trial, space))
        lr, it = params["learning_rate"], params["max_iter"]
        assert 0.02 <= lr <= 0.3
        assert (300 <= it <= 800) if lr < 0.06 else (100 <= it <= 500)
        assert it % 50 == 0 and params["class_weight"] in (None, "balanced")
        study.tell(trial, float(lr))


def test_unknown_parameter_type_is_rejected():
    with pytest.raises(ValueError):
        suggest(_study().ask(), {"x": {"type": "bogus"}})


def test_trial_table_labels_startup_and_adaptive_phases_and_failures():
    study = _study(n_startup=2)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)
    failed = study.ask()
    failed.set_user_attr("fail_reason", "boom")
    study.tell(failed, math.nan)
    table = trial_table(study, n_startup=2)
    assert list(table.phase) == ["startup_random", "startup_random", "adaptive_tpe", "adaptive_tpe"]
    assert table.state.iloc[-1] == "FAIL" and table.reason.iloc[-1] == "boom"
    assert pd.isna(table.objective.iloc[-1])
