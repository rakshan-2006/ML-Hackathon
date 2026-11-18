"""Seeded Optuna TPE (sequential model-based) search over one model family.

    python -m src.optimization.bayes_search                       # configs/optimisation.yaml
    python -m src.optimization.bayes_search --n-trials 2 --study smoke

Each trial fits the factory pipeline on the labelled rows of four canonical
folds and scores the fifth, for all five folds of the 80 train trips; the
objective is the configured mean validation metric. student_test is never
read here. The first n_startup_trials finished trials are TPE's seeded random
startup; later proposals come from TPE's density model of earlier results.
Every trial (complete, pruned or failed) is written to
results/optimization/<study>/trials.csv as it finishes; the Optuna study lives
in ignored artifacts/optimization/<study>.db so the search can be resumed.
"""

import argparse
import hashlib
import json
import math
import time
import traceback

import numpy as np
import optuna
import pandas as pd
import yaml
from threadpoolctl import threadpool_limits

from src.common.paths import PROJECT_ROOT
from src.common.schema import FEATURES
from src.evaluation.metrics import candidate_metrics, positive_proba
from src.models.model_factory import get_model, model_params
from src.training.cross_validation import N_SPLITS, fold_of_rows, folds_sha256, load_folds
from src.training.train_models import ARTIFACTS, DATA, SEED, THRESHOLD, load_partition, top1

CONFIG = PROJECT_ROOT / "configs" / "optimisation.yaml"
RESULTS = PROJECT_ROOT / "results" / "optimization"


def bind_study(study, contract):
    """Resume only the same experiment; never mix objectives/data/search spaces."""
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    previous = study.user_attrs.get('experiment_sha256')
    if previous is None:
        if study.trials:
            raise ValueError('Existing study has no experiment fingerprint; use a new study name')
        study.set_user_attr('experiment_sha256', fingerprint)
        study.set_user_attr('experiment_contract', contract)
    elif previous != fingerprint:
        raise ValueError('Study configuration/data/folds changed; use a new study name')
    return fingerprint


def suggest(trial, space):
    """Sample one parameter set; conditional entries get a branch-specific name."""
    params = {}
    for name, spec in space.items():
        kind = spec["type"]
        if "conditional_on" in spec:
            below = params[spec["conditional_on"]] < spec["below"]
            branch = spec["if_below"] if below else spec["otherwise"]
            tag = f"{name}|{spec['conditional_on']}<{spec['below']}" if below \
                else f"{name}|{spec['conditional_on']}>={spec['below']}"
            spec = {**spec, **branch}
        else:
            tag = name
        if kind == "float":
            params[name] = trial.suggest_float(tag, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "int":
            params[name] = trial.suggest_int(tag, spec["low"], spec["high"], step=spec.get("step", 1),
                                             log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(tag, spec["choices"])
        else:
            raise ValueError(f"unknown parameter type {kind!r} for {name}")
    return params


class FoldObjective:
    """Mean validation metric over the canonical folds, fitted inside each fold."""

    def __init__(self, model, metric, threads):
        self.model, self.metric, self.threads = model, metric, threads
        self.train = load_partition("train")
        self.folds = load_folds()
        cand = self.train["cand"]
        self.fold_all = fold_of_rows(cand.trip_id, self.folds)
        self.y_all = np.full(len(cand), -1)
        self.y_all[self.train["labelled"]] = self.train["y"]
        self.X = cand[FEATURES]

    def fold_scores(self, params, k):
        cand, lab = self.train["cand"], self.train["labelled"]
        tr = lab & (self.fold_all != k)
        va = self.fold_all == k
        if set(cand.trip_id[tr]) & set(cand.trip_id[va]):
            raise AssertionError(f"fold {k}: trip leakage")
        model = get_model(self.model, params, SEED)
        started = time.perf_counter()
        model.fit(self.X[tr], self.y_all[tr])
        fit_s = time.perf_counter() - started
        p_va = positive_proba(model, self.X[va])
        va_lab = va & lab
        scores = {**candidate_metrics(self.y_all[va_lab], p_va[lab[va]], THRESHOLD),
                  **top1(cand, p_va, va, self.train["obs"], self.train["align"],
                         trip_ids=self.folds.loc[self.folds.fold == k, "trip_id"])}
        return scores, fit_s

    def __call__(self, trial, space):
        params = suggest(trial, space)
        params = model_params(self.model, params)   # fills fixed defaults, rejects unknowns
        trial.set_user_attr("params", params)
        values, folds = [], []
        started = time.perf_counter()
        with threadpool_limits(self.threads):
            for k in range(N_SPLITS):
                scores, fit_s = self.fold_scores(params, k)
                values.append(scores[self.metric])
                folds.append({"fold": k, "fit_s": round(fit_s, 2),
                              **{m: scores[m] for m in ("f1_macro", "top1_accuracy", "log_loss",
                                                        "candidate_accuracy")}})
                trial.set_user_attr("folds", folds)
                trial.set_user_attr("elapsed_s", round(time.perf_counter() - started, 2))
                trial.report(scores[self.metric], k)
                if trial.should_prune():
                    trial.set_user_attr("prune_reason",
                                        f"fold {k} {self.metric}={scores[self.metric]:.4f} below the "
                                        f"median of earlier trials at that fold")
                    raise optuna.TrialPruned()
        for m in ("f1_macro", "top1_accuracy", "log_loss"):
            trial.set_user_attr(f"cv_mean_{m}", float(np.mean([f[m] for f in folds])))
        return float(np.mean(values))


def trial_table(study, n_startup):
    rows, finished_before = [], 0
    for t in sorted(study.trials, key=lambda t: t.number):
        phase = "startup_random" if finished_before < n_startup else "adaptive_tpe"
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            finished_before += 1
        folds = t.user_attrs.get("folds", [])
        rows.append({
            "trial": t.number, "state": t.state.name, "phase": phase,
            "objective": t.value, "elapsed_s": t.user_attrs.get("elapsed_s"),
            "folds_scored": len(folds),
            **{f"fold{f['fold']}_f1_macro": f["f1_macro"] for f in folds},
            "cv_mean_top1_accuracy": t.user_attrs.get("cv_mean_top1_accuracy"),
            "cv_mean_log_loss": t.user_attrs.get("cv_mean_log_loss"),
            "reason": t.user_attrs.get("prune_reason") or t.user_attrs.get("fail_reason"),
            "params": json.dumps(t.user_attrs.get("params", t.params), sort_keys=True),
            "fold_details": json.dumps(folds),
        })
    return pd.DataFrame(rows)


def plot_history(table, out, metric):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    done = table[table.state == "COMPLETE"]
    for phase, colour in (("startup_random", "#8a8a8a"), ("adaptive_tpe", "#2f6db5")):
        sub = done[done.phase == phase]
        ax.scatter(sub.trial, sub.objective, color=colour, label=f"{phase} (complete)", zorder=3)
    pruned = table[table.state == "PRUNED"]
    if len(pruned):
        partial = pruned.filter(like="_f1_macro").mean(axis=1)
        ax.scatter(pruned.trial, partial, marker="x", color="#b54a4a",
                   label="pruned (mean of scored folds)", zorder=3)
    best = done.set_index("trial").objective.reindex(table.trial).cummax().ffill()
    ax.step(table.trial, best, where="post", color="#3a9a5b", label="best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel(f"Mean 5-fold validation {metric}")
    ax.set_title(f"Optuna TPE search history (80 train trips, canonical folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "search_history.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--study", default=None, help="study name (default <model>_tpe)")
    parser.add_argument("--n-trials", type=int, default=None, help="override the configured budget")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    model, seed = cfg["model"], cfg["seed"]
    n_trials = args.n_trials or cfg["n_trials"]
    name = args.study or f"{model}_tpe"
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    db = ARTIFACTS / "optimization" / f"{name}.db"
    db.parent.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=cfg["n_startup_trials"],
                                         multivariate=True, group=True)
    prune = cfg["pruning"]
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=prune["n_startup_trials"],
                                          n_warmup_steps=prune["n_warmup_steps"])
              if prune["enabled"] else optuna.pruners.NopPruner())
    direction = cfg["objective"]["direction"]
    study = optuna.create_study(study_name=name, storage=f"sqlite:///{db.as_posix()}",
                                sampler=sampler, pruner=pruner, direction=direction,
                                load_if_exists=True)
    # The trial budget may grow on resume; all experiment semantics stay frozen.
    contract = {k: v for k, v in cfg.items() if k != 'n_trials'}
    contract.update(folds_sha256=folds_sha256(), dataset_manifest_sha256=
                    hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest())
    experiment_sha256 = bind_study(study, contract)
    objective = FoldObjective(model, cfg["objective"]["metric"], cfg["threads"])
    space = cfg["search_space"][model]

    def run_trial(trial):
        try:
            return objective(trial, space)
        except optuna.TrialPruned:
            raise
        except Exception as exc:   # recorded, then the study continues
            trial.set_user_attr("fail_reason", f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            return math.nan

    def save(study, trial):
        table = trial_table(study, cfg["n_startup_trials"])
        table.to_csv(out / "trials.csv", index=False)
        print(f"[{name}] trial {trial.number} {trial.state.name} value={trial.value} "
              f"elapsed={trial.user_attrs.get('elapsed_s')}s", flush=True)

    remaining = n_trials - len(study.trials)
    started = time.perf_counter()
    if remaining > 0:
        study.optimize(run_trial, n_trials=remaining, callbacks=[save])
    table = trial_table(study, cfg["n_startup_trials"])
    table.to_csv(out / "trials.csv", index=False)
    plot_history(table, out, cfg["objective"]["metric"])

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial if complete else None
    summary = {
        "method": "Optuna TPESampler (multivariate, grouped conditional space), seeded; "
                  "first n_startup_trials finished trials are random startup, later trials adaptive",
        "optuna_version": optuna.__version__, "model": model, "study": name, "seed": seed,
        "experiment_sha256": experiment_sha256,
        "configured_trials": cfg["n_trials"], "requested_trials": n_trials,
        "trials_by_state": table.state.value_counts().to_dict(),
        "trials_by_phase": table.phase.value_counts().to_dict(),
        "objective": cfg["objective"],
        "search_space": space,
        "folds_file": "configs/cv_folds.csv", "folds_sha256": folds_sha256(),
        "config_sha256": hashlib.sha256(open(args.config, "rb").read()).hexdigest(),
        "fitting_data": "labelled rows of the 80 train trips only; student_test not read",
        "threads": cfg["threads"],
        "total_trial_seconds": float(pd.to_numeric(table.elapsed_s, errors="coerce").sum()),
        "this_invocation_wall_seconds": round(time.perf_counter() - started, 1),
        "best_trial": best.number if best else None,
        "best_value": best.value if best else None,
        "best_params": best.user_attrs["params"] if best else None,
        "best_folds": best.user_attrs["folds"] if best else None,
        "selection_bias": "The best objective is the maximum over trials evaluated on the same "
                          "folds, so it is optimistically biased; it is not a nested-CV estimate.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("trials_by_state", "best_trial", "best_value",
                                              "best_params")}, indent=2, default=str))


if __name__ == "__main__":
    main()
