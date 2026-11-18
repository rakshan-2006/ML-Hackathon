# M1 traceability

Working baseline: origin/main (84abd85); no implemented consumers or repository
instructions. Interfaces are provisional pending M2/M3/M4 review, not sign-off.

| ID | Implementation | Evidence / acceptance |
|---|---|---|
| M1-01 | dataset_loader.py | test_data_loader.py; inventory.csv |
| M1-02 | eda.py | inventory.csv; data_quality_log.md |
| M1-03 | geo.py | projection_errors.csv; projection tests |
| M1-04 | track_cleaning.py | C1 validation; quality log |
| M1-05 | road_graph.py | graph_stats.csv; graph tests |
| M1-06 | candidate_generation.py | candidate_sweep.csv; candidate tests |
| M1-07 | reference_alignment.py | alignment_stats.csv; maps; tests |
| M1-08 | feature_engineering.py | coverage.csv; label tests |
| M1-09 | feature_engineering.py | feature schema; leakage tests |
| M1-10 | common/schema.py | full validation; dtype tests |
| M1-11 | build_dataset.py | manifest; reproducibility.json |
| M1-12 | eda.py; folium_maps.py | figures/eda; maps/eda_* |
| M1-13 | tests/fixtures; tests | pytest result |
| M1-14 | m1_data_features.md; m1_handoff.md | measured report and commands |

Status: implementation in progress. Final acceptance recorded in report.
