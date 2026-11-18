"""Wiring/latency smoke test on 20,000 fit and 5,000 held-out-trip candidate rows.

This does not publish accuracy results, fit the final model or replace full CV.
"""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from src.training.train_models import main as train
    train(['--model', 'knn', '--smoke-only', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
