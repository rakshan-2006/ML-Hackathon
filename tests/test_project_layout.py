"""Integration checks for canonical config, artifact paths and model contracts."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.common.paths import PROJECT_ROOT
from src.common.schema import FEATURES, SCHEMA_VERSION
from src.training.cross_validation import FOLDS_PATH, load_folds
from src.training.train_gradient_boosting import make_folds


def test_canonical_folds_keep_original_membership_and_bytes():
    folds = load_folds()
    card = json.loads((PROJECT_ROOT/'results/models/logistic_regression/model_card.json').read_text())
    assert hashlib.sha256(FOLDS_PATH.read_bytes()).hexdigest() == card['folds_sha256']
    groups = folds.trip_id.copy()
    mapping, assigned, _ = make_folds(groups, FOLDS_PATH)
    assert mapping == dict(zip(folds.trip_id.astype(str), folds.fold))
    assert np.array_equal(assigned, folds.fold)
    assert FOLDS_PATH == PROJECT_ROOT/'configs/cv_folds.csv'


def test_dependency_sets_exist_and_have_no_conflicting_pins():
    visited, pins = set(), {}
    def visit(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('-r '):
                visit(path.parent/line[3:])
            else:
                package, version = line.split('==')
                package = package.lower()
                assert pins.setdefault(package, version) == version
    visit(PROJECT_ROOT/'requirements.txt')
    assert {'scikit-learn', 'fastapi', 'folium', 'pytest'} <= set(pins)


def test_archived_json_fold_maps_still_load(tmp_path):
    folds = load_folds()
    path = tmp_path/'legacy.json'
    path.write_text(json.dumps({'trip_to_fold':dict(zip(folds.trip_id.astype(str),folds.fold.astype(int)))}))
    pd.testing.assert_frame_equal(load_folds(path).reset_index(drop=True),folds)


def test_documentation_relative_links_resolve():
    documents = [PROJECT_ROOT/'README.md', PROJECT_ROOT/'data/README.md']
    documents += list((PROJECT_ROOT/'docs').rglob('*.md'))
    documents += [PROJECT_ROOT/'results/models/gradient_boosting/figures/README.md']
    for document in documents:
        for href in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8-sig')):
            href = href.split('#')[0]
            if href and '://' not in href:
                assert (document.parent/href).exists(), (document, href)


def test_comparison_reads_all_three_existing_baseline_formats():
    from src.evaluation.compare_models import comparison_table
    names = ['logistic_regression','decision_tree','gradient_boosting']
    table = comparison_table(names).set_index('model')
    assert table.index.tolist() == names
    card = json.loads((PROJECT_ROOT/'results/models/gradient_boosting/model_card.json').read_text())
    assert np.isclose(table.loc['gradient_boosting','cv_f1_macro'],card['cv_fold_mean']['macro_f1'])
    assert np.isclose(table.loc['gradient_boosting','student_test_top1_accuracy'],card['student_test']['top1_accuracy'])


@pytest.mark.parametrize('model_name,params', [
    ('gradient_boosting', {'max_iter': 2, 'min_samples_leaf': 2}),
    ('random_forest', {'n_estimators': 3, 'max_depth': 3, 'n_jobs': 1}),
    ('knn', {'n_neighbors': 3, 'n_jobs': 1}),
])
def test_shared_runner_trains_and_separates_artifacts(tmp_path, monkeypatch, model_name, params):
    from src.training import train_models as runner
    from src.evaluation import visualizations
    data, results, artifacts = tmp_path/'data', tmp_path/'results', tmp_path/'artifacts'
    data.mkdir()
    fold_table = pd.DataFrame({'trip_id': [f'{i:08d}' for i in range(10)],
                               'fold': [i%5 for i in range(10)]})
    folds_path = tmp_path/'folds.csv'
    fold_table.to_csv(folds_path, index=False)
    for partition, ids in [('train', range(10)), ('student_test', [99])]:
        rows, obs, align = [], [], []
        for trip in ids:
            tid = f'{trip:08d}'
            for idx in range(5):
                obs.append(dict(trip_id=tid, obs_idx=idx, is_valid=True))
                align.append(dict(trip_id=tid, obs_idx=idx, align_status='ok', ref_arc_id=0))
                for arc in [0,1]:
                    row = dict.fromkeys(FEATURES, 0.)
                    row.update(trip_id=tid, obs_idx=idx, arc_id=arc, cand_rank=arc,
                               dist_m=float(arc*10), n_candidates=2, label=1-arc,
                               heading_diff_deg=np.nan if idx==0 else float(arc*90))
                    rows.append(row)
        pd.DataFrame(rows).to_parquet(data/f'{partition}_candidates.parquet',index=False)
        pd.DataFrame(obs).to_parquet(data/f'{partition}_observations.parquet',index=False)
        pd.DataFrame(align).to_parquet(data/f'{partition}_alignment.parquet',index=False)
    (data/'manifest.json').write_text(json.dumps({'schema_version':SCHEMA_VERSION}))
    for name, value in [('DATA',data),('RESULTS',results),('ARTIFACTS',artifacts),
                        ('FOLDS_PATH',folds_path),('PROJECT_ROOT',tmp_path)]:
        monkeypatch.setattr(runner,name,value)
    monkeypatch.setattr(runner,'load_folds',lambda:fold_table)
    monkeypatch.setattr(runner,'folds_sha256',lambda:hashlib.sha256(folds_path.read_bytes()).hexdigest())
    monkeypatch.setattr(runner,'smoke_test',lambda *a,**kw:{'fixture_smoke':True})
    monkeypatch.setattr(visualizations,'plot_model_accuracy',lambda name:None)
    with threadpool_limits(limits=2):
        runner.run(model_name, params)
    assert (artifacts/model_name/'pipeline.joblib').is_file()
    assert not list(results.rglob('*.joblib'))
    assert not list(results.rglob('*.parquet'))
    card = json.loads((results/model_name/'model_card.json').read_text())
    assert card['row_counts']['train_trips'] == 10
    if model_name == 'gradient_boosting':
        assert card['preprocessing']['missing_values'].startswith('Native')
    else:
        assert card['preprocessing']['missing_values'].startswith('SimpleImputer')
    scores = pd.read_parquet(artifacts/model_name/'oof_probabilities.parquet')
    assert len(scores) == 100 and scores.p_model.between(0,1).all()
