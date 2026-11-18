# Review figures: baseline, tuning and HMM

Generated with `python -m src.visualization.review_comparison`. PNG and SVG versions are supplied.

1. [Three-stage GPS Top-1 comparison](01_baseline_tuned_hmm_top1.png): baseline GB **83.57%**, tuned GB **83.74%**, selected baseline GB + HMM **91.45%**.
2. [Before/after tuning](02_before_after_tuning.png): training-CV selection objective alongside student-test Top-1, macro-F1 and log loss.
3. [HMM per trip](03_hmm_per_trip.png): paired classifier/HMM scores across all ten student-test trips.

**Interpretation:** The tuned trial slightly improves test Top-1 but does not beat baseline training-CV macro-F1, so the baseline remains selected. HMM was applied to that selected baseline, not to the tuned trial. Its gain is **+7.88 percentage points over baseline**. No tuned-model-plus-HMM result is claimed.

All Top-1 scores use 16,073 trusted GPS observations; candidate misses count as failures. The aggregate HMM score is weighted by observations. Fixed untuned decoder defaults are used, and trip 00000007 worsens.

Evidence: [baseline](../../models/gradient_boosting/student_test_metrics.csv), [tuned](../../models/gradient_boosting_tuned/student_test_metrics.csv), [HMM summary](../../rehearsal/summary.json), [per-trip data](../../rehearsal/student_test_api.csv). Exact plotting values are in [figure_data.json](figure_data.json) and [stage_top1.csv](stage_top1.csv).
