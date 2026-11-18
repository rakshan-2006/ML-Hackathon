"""Log-space Viterbi decoding over a candidate lattice."""

import numpy as np


def viterbi(log_em, log_tr):
    """Most likely state sequence for one chain of observations.

    log_em : list of T 1-D arrays; log_em[t][k] = log emission of candidate k at step t.
    log_tr : list of T-1 2-D arrays; log_tr[t-1][i, j] = log transition from candidate i
             at step t-1 to candidate j at step t (-inf = impossible).

    Returns (path, score, break_at):
      path     : best candidate index for each step in [0, break_at)
      score    : log probability of that path
      break_at : T if the whole chain was decoded; otherwise the first step t that no
                 finite-probability path can reach (caller restarts decoding at t).
    Ties are broken towards the lower candidate index.
    """
    T = len(log_em)
    if T == 0:
        return [], 0.0, 0
    if len(log_tr) != T - 1:
        raise ValueError("need exactly one transition matrix per consecutive step pair")

    delta = np.asarray(log_em[0], dtype=np.float64)
    if delta.size == 0 or not np.isfinite(delta).any():
        raise ValueError("first step has no candidate with finite emission")
    backptrs = []
    for t in range(1, T):
        em = np.asarray(log_em[t], dtype=np.float64)
        tr = np.asarray(log_tr[t - 1], dtype=np.float64)
        if tr.shape != (delta.size, em.size):
            raise ValueError(f"transition matrix {t - 1} has shape {tr.shape}, "
                             f"expected {(delta.size, em.size)}")
        scores = delta[:, None] + tr
        bp = np.argmax(scores, axis=0)
        new_delta = scores[bp, np.arange(em.size)] + em
        if not np.isfinite(new_delta).any():
            return _backtrack(delta, backptrs), float(delta.max()), t
        delta = new_delta
        backptrs.append(bp)
    return _backtrack(delta, backptrs), float(delta.max()), T


def _backtrack(delta, backptrs):
    state = int(np.argmax(delta))
    path = [state]
    for bp in reversed(backptrs):
        state = int(bp[state])
        path.append(state)
    return path[::-1]
