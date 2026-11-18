"""Server-only evaluation extension point.

Reference-route evaluation is intentionally unimplemented. Inference and API
responses must never read or expose reference `.route` files.
"""

from typing import Any


def evaluate_prediction(*args: Any, **kwargs: Any) -> None:
    """Placeholder for future server-side evaluation only."""
    raise NotImplementedError(
        "Evaluation is not implemented until the evaluator contract is agreed."
    )
