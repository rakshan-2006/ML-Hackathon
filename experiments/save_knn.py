"""Compatibility command for full five-fold KNN evaluation and final model saving.

Prefer: python -m src.training.train_models --model knn
"""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    from src.training.train_models import main as train
    train(['--model', 'knn', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
