"""Compatibility entry point; Random Forest uses the shared five-fold trainer.

Preferred: python -m src.training.train_models --model random_forest
"""
import argparse
import json
from src.training.train_models import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', default='{}', help='JSON parameter overrides')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--folds', type=int, choices=[5], default=5,
                        help='Only the canonical five trip-grouped folds are supported')
    args = parser.parse_args(argv)
    run('random_forest', json.loads(args.params), args.smoke_only)


if __name__ == '__main__':
    main()
