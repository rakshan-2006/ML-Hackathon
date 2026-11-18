# Person A / Person B integration contract

Person B owns decoder, route metrics, pipeline, API/UI and deployment. Person A
owns `src/inference/predictor.py`, the model/scorer implementation and final score
exports. This document records an implementation contract, not invented approval.

## Scoring

`Predictor.load(models_dir)` returns an object with
`score_candidates(cand_df) -> np.ndarray[float64]`. Preserve row order, accept empty
tables, select class 1 using `classes_`, apply preprocessing exactly once and reject
schema/feature mismatches. C2 metadata/label columns never become features.

Deployment filenames: `best_model.joblib` (estimator), `preprocessing.joblib`
(fitted transform), `model_card.json` (artifact_layout, model_version, schema_version,
ordered_features), and B's `decoder_config.json`. All final model files must be
bound to their checksums. B's explicit development adapter loads the verified GB
Release pipeline; it is never presented as Person A's final selected/tuned model.

## Decoder and metrics

`run_trip(trip_dir, score_candidates, data_config, decoder_config)` is route-free.
It returns C1/C2, same-order scores, one prediction per raw observation, segment
IDs, raw/assembled routes, geometry and millisecond timings. `decode_tables` adapts
M1 tables to the existing tested list-level HMM. Prediction code never reads route
references; `src/api/evaluation.py` calls isolated server-only evaluation afterwards.

`src.hmm.route_metrics` is B's single directed route-metric implementation.
`pointwise_accuracy` counts missing predictions against trusted references as wrong.
`route_continuity(arcs, graph, segment_ids=None)` excludes gap/missing boundaries,
collapses consecutive repeats, and reports counts plus the connected fraction.
Pointwise continuity is compared using the same GPS-gap segmentation for both
classifier-only and HMM. Assembled continuity is separate because path insertion
and decoder restarts can improve it mechanically. Report break/inferred counts.
`sequence_similarity` reports set arc overlap and exact order-aware LCS F1.

Undefined internal metrics may be NaN; the public JSON boundary converts them to
null. Aggregate by numerators/denominators rather than averaging trip scores blindly.

## Final-model gate

Final decoder tuning/results and final report assembly wait for A's selected model,
final-parameter train OOF and student-test exports, all five baseline comparison,
Bayesian search evidence and report contribution. Required C4 keys are trip_id,
obs_idx, arc_id, p_model, model_name, model_version; cover ALL candidates, not only
trusted-label rows. Record dataset/fold/config hashes and explicit OOF provenance.

Development may use the versioned GB baseline and fixed **untuned** decoder config.
All such results are labelled development-baseline, never final evaluation.
The evaluator archive stays sealed. Two-laptop hardware validation requires an
actual second machine and must not be inferred from local TestClient checks.

## Final OOF provenance manifest consumed by `src.hmm.tune`

JSON fields: `status=final_parameter_oof`, `schema_version`, `model_version`,
`partition=train`, `score_kind=out_of_fold`, `fold_holdout_trip_ids` (string fold
number -> complete canonical held-out trip ID list), and `sha256` with keys
`candidates`, `observations`, `alignment`, `scores`, `model_card`, `folds`,
`data_config`. Hash the exact consumed file bytes. A must document how the export
was generated using the final selected classifier parameters with fold-local
preprocessing and no held-out trip fitting. A manifest is declared provenance,
not proof replacing review of the producing code/training record.

The C4 export additionally requires `fold` for every candidate. IDs remain strings
(preserve leading zeros). B validates all key identities including unlabeled rows,
canonical folds, positive-class probabilities and the final card/version before
creating a tuning run. Missing evidence is an error, never implicit permission to
tune in-sample probabilities. These fields are proposed integration requirements;
they do not imply Person A has already supplied/approved them.
