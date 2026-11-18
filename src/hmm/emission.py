"""Emission scores from Member 2's candidate probabilities.

The classifier gives P(candidate is the true arc | features). We use its log as the
emission evidence for that hidden state (a standard approximation: it is a posterior,
not a likelihood P(observation | arc)). No probabilities are produced here.
"""

import numpy as np


def log_emissions(probs, eps=1e-6, weight=1.0):
    """Convert one observation's candidate probabilities into log-emission scores.

    probs  : 1-D sequence, one probability per candidate (same order as the candidates).
    eps    : floor applied before the log so p == 0 gives a finite, very low score.
    weight : scales emissions relative to transitions (decoder parameter).
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError("probs must be 1-D (one value per candidate)")
    if not np.isfinite(p).all():
        raise ValueError("candidate probabilities contain NaN or infinity")
    if not 0 < eps < 1 or not np.isfinite(weight) or weight < 0:
        raise ValueError('emission eps must lie in (0,1), weight must be finite and nonnegative')
    if (p < 0).any() or (p > 1).any():
        raise ValueError("candidate probabilities must lie in [0, 1]")
    return weight * np.log(np.clip(p, eps, 1.0))
