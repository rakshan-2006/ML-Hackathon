| Model | Algorithm Family | Key Configuration | CV Macro F1-score (mean of 5 folds) | Top-1 Accuracy (student-test observations) | Route continuity (student-test, classifier-only) | Student-Test Macro F1-score | Log Loss (student-test) | Inference Time (µs per candidate, student-test) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear and probabilistic | C=1.0, class_weight=None, max_iter=1000 | 0.8672 ± 0.0331 | 0.8279 | 0.7446 | 0.8895 | 0.1526 | 0.71 |
| Decision Tree | Tree-based | max_depth=12, min_samples_leaf=50, class_weight=None | 0.8759 ± 0.0398 | 0.8316 | 0.7403 | 0.8983 | 0.1351 | 0.49 |
| Random Forest | Ensemble | n_estimators=80, max_depth=24, min_samples_leaf=2, max_features=0.5, class_weight=balanced | 0.8818 ± 0.0262 | 0.8377 | 0.7458 | 0.8946 | 0.1471 | 4.48 |
| HistGradientBoosting | Boosting | learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0 | 0.8861 ± 0.0312 | 0.8357 | 0.7545 | 0.9039 | 0.1180 | 18.19 |
| KNN | Instance-based | n_neighbors=5, weights=distance, metric=minkowski, p=2, algorithm=kd_tree | 0.8579 ± 0.0361 | 0.8102 | 0.7152 | 0.8848 | 0.7103 | 1872.41 |
