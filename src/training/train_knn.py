"""Full KNN evaluation with bounded queries and exact training-only diagnostics.

Run: python -m src.training.train_knn --jobs 8
Validation and student-test candidates always use ordinary KNN queries.
"""
import time
import numpy as np
import pandas as pd
from src.training import train_models
from src.evaluation.metrics import positive_proba

call = 0

def batched_proba(model, X):
    global call
    call += 1
    label, start = call, time.perf_counter()
    estimator = model.named_steps['model']
    if estimator.weights == 'distance' and set(estimator.classes_) == {0, 1} and len(X) == estimator.n_samples_fit_:
        transformed = model[:-1].transform(X)
        if np.array_equal(transformed, estimator._fit_X):
            # At distance zero, distance weighting ignores all nonzero-distance
            # neighbours. If every identical training vector has the same label,
            # its probability is exactly that label. Hash collisions only send
            # extra mixed groups to the ordinary query path. Normalize signed 0.
            hashes = pd.util.hash_pandas_object(
                pd.DataFrame(np.where(transformed == 0, 0., transformed)), index=False)
            targets = pd.Series(estimator.classes_[estimator._y])
            minimum = targets.groupby(hashes).transform('min').to_numpy()
            maximum = targets.groupby(hashes).transform('max').to_numpy()
            output = targets.to_numpy(dtype=np.float64)
            mixed = np.flatnonzero(minimum != maximum)
            for first in range(0, len(mixed), 10000):
                idx = mixed[first:first+10000]
                output[idx] = positive_proba(model, X.iloc[idx])
            check = np.random.default_rng(2026).choice(len(X), min(256, len(X)), replace=False)
            assert np.allclose(output[check], positive_proba(model, X.iloc[check]), rtol=0, atol=1e-12)
            print(f'[KNN training diagnostics] {len(X):,} exact self-matches; '
                  f'{len(mixed):,} mixed-label rows queried normally; verified {len(check)} queries; '
                  f'{time.perf_counter()-start:.1f}s', flush=True)
            return output
    output = np.empty(len(X), dtype=np.float64)
    print(f'[KNN scoring {label}] {len(X):,} held-out rows', flush=True)
    for first in range(0, len(X), 10000):
        end = min(first + 10000, len(X))
        output[first:end] = positive_proba(model, X.iloc[first:end])
        if end % 50000 == 0 or end == len(X):
            print(f'[KNN scoring {label}] {end:,}/{len(X):,}; {time.perf_counter()-start:.1f}s', flush=True)
    return output

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    original = train_models.positive_proba
    global call
    call = 0
    try:
        train_models.positive_proba = batched_proba
        train_models.run('knn', {'n_jobs': args.jobs}, smoke_only=args.smoke_only)
    finally:
        train_models.positive_proba = original


if __name__ == '__main__':
    main()
