Classifier-only vs classifier+HMM on 10 student_test trips (model hgb-baseline-ac522dfa, decoder development-untuned-v1; development measurements, not evaluator results). Source: `results/rehearsal/student_test_api.csv`.

| Metric | Classifier only | Classifier + HMM | Aggregation |
|---|---:|---:|---|
| Top-1 accuracy (trusted observations) | 83.57% | 91.45% | sum n_correct / sum n_evaluated |
| Route continuity (connected / transitions) | 75.45% | 88.47% | sum n_connected_transitions / sum n_continuity_transitions |
| Route breaks | 1,709 | 409 | sum n_route_breaks |
| Arc-overlap F1 vs reference route | 0.798 | 0.945 | mean of per-trip arc_f1 |
| Order-aware LCS similarity vs reference route | 0.610 | 0.902 | mean of per-trip lcs_similarity |

HMM Top-1 better on 9 trips, worse on 1 (00000007).
