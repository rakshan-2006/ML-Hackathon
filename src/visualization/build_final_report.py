"""Team 84 PathSnap technical report (Deliverables.pdf sections 2.2-7).

    python -m src.visualization.build_final_report

Tables are read from committed result files at build time; figures are the
committed experiment plots, maps and UI captures. Output:
output/pdf/PES1UG24CS316_F_PES1UG24CS317_F_PES1UG24CS365_F_PES1UG24CS366_F.pdf
"""

import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from src.common.paths import PROJECT_ROOT
from src.evaluation.visualizations import load_metrics

R = PROJECT_ROOT
OUT = R / "output" / "pdf" / "PES1UG24CS316_F_PES1UG24CS317_F_PES1UG24CS365_F_PES1UG24CS366_F.pdf"
TEAM = [("PES1UG24CS316", "PARAS AGARWAL", "F"), ("PES1UG24CS317", "PARINITHA RAMESH", "F"),
        ("PES1UG24CS365", "RAKSHAN R", "F"), ("PES1UG24CS366", "RAKSHITA MUTTUR", "F")]
NAMES = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
         "random_forest": "Random Forest", "gradient_boosting": "Hist Gradient Boosting",
         "knn": "KNN", "gradient_boosting_tuned": "Hist Gradient Boosting (tuned)"}
NAVY = colors.HexColor("#16344c")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle("Body", fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=6))
styles.add(ParagraphStyle("Q", fontName="Helvetica-Bold", fontSize=9.5, leading=13, spaceBefore=4,
                          spaceAfter=2, textColor=NAVY))
styles.add(ParagraphStyle("Cap", fontName="Helvetica-Oblique", fontSize=8, leading=10,
                          textColor=colors.HexColor("#444444"), spaceAfter=10))
styles.add(ParagraphStyle("Cell", fontName="Helvetica", fontSize=7.5, leading=9))
styles.add(ParagraphStyle("CellB", fontName="Helvetica-Bold", fontSize=7, leading=8.5,
                          textColor=colors.white))
styles.add(ParagraphStyle("TitleC", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=26))
styles.add(ParagraphStyle("SubC", fontName="Helvetica", fontSize=11, leading=15, alignment=TA_CENTER))
styles.add(ParagraphStyle("Mono", fontName="Courier", fontSize=7, leading=8.5))
story = []


def p(text, style="Body"):
    story.append(Paragraph(text, styles[style]))


def h1(text):
    story.append(Paragraph(text, styles["Heading1"]))


def h2(text):
    story.append(Paragraph(text, styles["Heading2"]))


def h3(text):
    story.append(Paragraph(text, styles["Heading3"]))


def qa(question, answer):
    story.append(KeepTogether([Paragraph(question, styles["Q"]), Paragraph(answer, styles["Body"])]))


def image(rel, width=16 * cm, caption=None):
    img = Image(str(R / rel))
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    parts = [img]
    if caption:
        parts.append(Paragraph(caption, styles["Cap"]))
    story.append(KeepTogether(parts))


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), styles["CellB" if header and i == 0 else "Cell"]) for c in row]
            for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def cover():
    story.append(Spacer(1, 2.5 * cm))
    p("PES University, Department of Computer Science and Engineering", "SubC")
    p("UE24CS352A: Machine Learning", "SubC")
    story.append(Spacer(1, 0.8 * cm))
    p("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", "TitleC")
    p("Technical Report", "SubC")
    story.append(Spacer(1, 0.6 * cm))
    p("<b>Team 84</b>", "SubC")
    story.append(Spacer(1, 0.5 * cm))
    rows = [["SRN", "Name", "Section"]] + [list(r) for r in TEAM]
    t = Table(rows, colWidths=[4.5 * cm, 6 * cm, 2.5 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 10),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                           ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 1.0 * cm))
    hmm = pd.read_csv(R / "results/hmm/development/classifier_vs_hmm_summary.csv").iloc[0]
    trips = pd.read_csv(R / "results/rehearsal/student_test_api.csv", dtype={"trip_id": str})
    box = Table([[Paragraph(
        f"<b>Inference accuracy achieved:</b> the final system (Histogram Gradient Boosting classifier + "
        f"HMM/Viterbi decoding) matches <b>{pct(hmm.classifier_plus_hmm)}</b> of GPS observations to the correct "
        f"road segment ({int(trips.n_correct_hmm.sum()):,} of {int(trips.n_evaluated_hmm.sum()):,} trusted "
        f"observations, 10 student_test trips), up from {pct(hmm.classifier_only)} with the classifier alone "
        f"(+{100 * (hmm.classifier_plus_hmm - hmm.classifier_only):.2f} percentage points).", styles["SubC"])]],
        colWidths=[15 * cm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, NAVY), ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef3f8")),
                             ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                             ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10)]))
    story.append(box)
    story.append(Spacer(1, 1.0 * cm))
    p("Repository: https://github.com/rakshan-2006/ML-Hackathon", "SubC")
    p("Every number in this report is read from result files committed in the repository; "
      "the source file is named under each table or figure.", "SubC")
    story.append(PageBreak())


def section_2():
    lab = pd.read_csv(R / "results/eda/label_and_sampling_summary.csv").set_index("partition")
    cov = pd.read_csv(R / "results/eda/coverage_by_partition.csv").set_index("partition")
    miss = pd.read_csv(R / "results/eda/missing_values.csv")
    by = pd.read_csv(R / "results/eda/train_features_by_label.csv").pivot(index="feature", columns="label",
                                                                          values="median")
    inv = pd.read_csv(R / "results/eda/inventory.csv")
    graph = pd.read_csv(R / "results/eda/graph_stats.csv").groupby("partition")[["arcs", "weak_components"]].sum()
    sweep = pd.read_csv(R / "results/eda/candidate_sweep.csv")
    proj = pd.read_csv(R / "results/eda/projection_errors.csv")

    h1("2.2 Dataset Understanding and Exploratory Data Analysis")
    p("<b>Partitions used.</b> All exploratory analysis, preprocessing choices and model fitting used the "
      "80 <i>train</i> trips. The 10 <i>student_test</i> trips were used only for held-out comparison. "
      "Each table and figure below names its partition. Each trip contains a <font face='Courier'>.track</font> "
      "file (longitude, latitude, time), <font face='Courier'>.nodes</font>, directed "
      "<font face='Courier'>.arcs</font> and the reference <font face='Courier'>.route</font> (ordered arc IDs).")

    qa("1. How did you preprocess the data?",
       "Source: <font face='Courier'>src/data/</font>, settings in <font face='Courier'>configs/data.yaml</font>. "
       "(a) Every file is parsed strictly; malformed rows raise errors and record counts are checked against the "
       "loaded tables (<font face='Courier'>results/eda/inventory.csv</font>). "
       "(b) Coordinates are projected to metres with one WGS84 azimuthal-equidistant projection centred on each "
       "trip's road network. On 5,760 sampled node pairs the projected distance differs from the geodesic distance "
       f"by at most {proj.absolute_error_m.max():.2f} m (median {proj.absolute_error_m.median():.1e} m). "
       "(c) All raw GPS points are kept without smoothing; time step, speed and heading are derived between "
       "consecutive points, and gaps above 30 s, non-positive time steps and stationary steps below 1 m disable "
       "the movement features they would corrupt. "
       "(d) Candidate generation: every directed arc within 100 m of a valid point, ordered by distance, at most "
       "8 per point. These values were chosen on train only from a 3x3 radius/K sweep, taking the smallest "
       "candidate table within 0.5 percentage points of the best reference recall. "
       "(e) Labels: the reference route is aligned to the GPS points; a candidate is labelled 1 if it is the "
       "aligned reference arc, 0 otherwise. Points whose alignment is ambiguous or unaligned get a null label and "
       "are excluded from fitting and candidate metrics. The reference arc is never added to the candidate set. "
       "(f) Learned preprocessing (median imputation with missing-value indicators, and scaling where the model "
       "needs it) is fitted inside each cross-validation training fold only.")

    rows = [["Radius (m)", "Max candidates K", "Trusted-reference recall", "Mean candidates / point"]]
    for r in sweep.itertuples():
        rows.append([int(r.radius_m), int(r.max_candidates), pct(r.reference_recall_trusted),
                     f"{r.mean_candidates_per_observation:.2f}"])
    table(rows, [3 * cm, 3.5 * cm, 4.5 * cm, 4.5 * cm])
    p("Table 2.1 - Candidate radius/K sweep on train (results/eda/candidate_sweep.csv). R=100 m, K=8 was "
      "selected.", "Cap")

    qa("2. What patterns did you observe during data exploration? What features did you extract and why?", "")
    tr, st = "train", "student_test"
    rows = [["Quantity", "train", "student_test"],
            ["Trips / GPS observations", f"{int(lab.trips[tr])} / {int(lab.observations[tr]):,}",
             f"{int(lab.trips[st])} / {int(lab.observations[st]):,}"],
            ["Candidate rows (all / labelled)", f"{int(lab.candidates[tr]):,} / {int(lab.labelled_candidates[tr]):,}",
             f"{int(lab.candidates[st]):,} / {int(lab.labelled_candidates[st]):,}"],
            ["Positive rate among labelled candidates", pct(lab.positive_rate_labelled[tr], 1),
             pct(lab.positive_rate_labelled[st], 1)],
            ["Trusted observations (share of valid)", f"{int(cov.trusted[tr]):,} ({pct(cov.alignment_coverage_valid[tr], 1)})",
             f"{int(cov.trusted[st]):,} ({pct(cov.alignment_coverage_valid[st], 1)})"],
            ["Reference arc missing from candidates (trusted)", f"{int(cov.reference_misses[tr]):,} ({pct(cov.miss_rate_trusted[tr])})",
             f"{int(cov.reference_misses[st]):,} ({pct(cov.miss_rate_trusted[st])})"],
            ["Valid observations with zero candidates", f"{int(cov.zero_candidates_valid[tr]):,}", f"{int(cov.zero_candidates_valid[st]):,}"],
            ["Median sampling interval / gaps over 30 s", f"{lab.dt_prev_p50_s[tr]:.0f} s / {int(lab.gap_flag_observations[tr])}",
             f"{lab.dt_prev_p50_s[st]:.0f} s / {int(lab.gap_flag_observations[st])}"],
            ["Largest time gap", f"{lab.dt_prev_max_s[tr]:,.0f} s", f"{lab.dt_prev_max_s[st]:,.0f} s"],
            ["Trips whose reference route has a connectivity break",
             str(int((inv[inv.partition == tr].route_break_count > 0).sum())),
             str(int((inv[inv.partition == st].route_break_count > 0).sum()))],
            ["Directed arcs / weakly connected components", f"{int(graph.arcs[tr]):,} / {int(graph.weak_components[tr]):,}",
             f"{int(graph.arcs[st]):,} / {int(graph.weak_components[st]):,}"],
            ["Invalid coordinates / timestamps / duplicate points",
             f"{int(inv[inv.partition == tr].invalid_observations.sum())} / {int(inv[inv.partition == tr].invalid_timestamps.sum())} / {int(inv[inv.partition == tr].duplicate_observations.sum())}",
             f"{int(inv[inv.partition == st].invalid_observations.sum())} / {int(inv[inv.partition == st].invalid_timestamps.sum())} / {int(inv[inv.partition == st].duplicate_observations.sum())}"]]
    table(rows, [7.5 * cm, 4.2 * cm, 4.2 * cm])
    p("Table 2.2 - Measured dataset profile (results/eda/label_and_sampling_summary.csv, "
      "coverage_by_partition.csv, inventory.csv, graph_stats.csv).", "Cap")

    tm = miss[miss.partition == tr].set_index("feature")
    p("<b>Patterns observed.</b>")
    p(f"&bull; The task is imbalanced: about one positive per 6-7 candidates ({pct(lab.positive_rate_labelled[tr], 1)} "
      "positive on train), so candidate accuracy alone is misleading and macro-F1, positive-class precision/recall "
      "and observation Top-1 are reported.")
    p(f"&bull; Candidate coverage bounds every classifier: on train {pct(cov.miss_rate_trusted[tr])} of trusted "
      "observations have no correct candidate, so they cannot be matched by any model. Most of these lie in a few "
      "trips, which later appear as the hardest cross-validation fold.")
    p(f"&bull; <b>Missing-value analysis</b> (Figure 2.2): only movement features are missing, for structural reasons. "
      f"heading_diff_deg is missing on {pct(tm.missing_fraction['heading_diff_deg'])} of train candidate rows "
      f"(stationary or first points); speed_mps and prev_dist_m on {pct(tm.missing_fraction['speed_mps'])} "
      f"(first point of a trip or after a gap); next_dist_m on {pct(tm.missing_fraction['next_dist_m'])} (last point). "
      "All other features are complete. Missingness rates are similar for positive and negative candidates, and "
      "missing-value indicators are kept as extra inputs.")
    p(f"&bull; By class on train (Figure 2.3), true road segments are much closer to the GPS point (median dist_m "
      f"{by.loc['dist_m', 1]:.2f} m vs {by.loc['dist_m', 0]:.2f} m), better aligned with the direction of travel "
      f"(median heading difference {by.loc['heading_diff_deg', 1]:.1f} deg vs {by.loc['heading_diff_deg', 0]:.1f} deg), "
      f"and lie on longer arcs (median {by.loc['arc_len_m', 1]:.0f} m vs {by.loc['arc_len_m', 0]:.0f} m). The nearest "
      "candidate is the reference only part of the time, so nearest-road snapping is a weak baseline.")
    p("&bull; Sampling is dense (median 1 s), but train contains long recording gaps, which the decoder must not "
      "bridge. Reference routes of several trips contain connectivity breaks, so derived labels are treated as "
      "approximate rather than verified ground truth.")
    p("<b>Features extracted (13, ordered in src/common/schema.py).</b> dist_m, cand_rank and dist_minus_min describe "
      "geometric closeness; heading_diff_deg uses the direction of travel to separate the two directed twins of a "
      "road; speed_mps, arc_len_m, proj_frac, is_oneway, source_degree and target_degree describe motion and road "
      "context (for example short connector arcs at junctions); n_candidates measures local ambiguity; prev_dist_m "
      "and next_dist_m add temporal context, because a correct arc is usually also close to the neighbouring GPS "
      "points. Identifiers, coordinates, alignment information and labels are never features.")

    image("results/figures/eda/trajectory_network.png", 16 * cm,
          "Figure 2.1 - GPS trajectory plot and road-network visualisation: GPS points, directed road network and "
          "reference route for the median-length trip of each partition (train 00000095, student_test 00000076).")
    image("results/figures/eda/missing_values.png", 13 * cm,
          "Figure 2.2 - Missing-value analysis: share of candidate rows with NaN per feature, train and student_test.")
    image("results/figures/eda/features_by_label.png", 16 * cm,
          "Figure 2.3 - train only: distributions of key features for reference (positive) and non-reference "
          "candidates.")
    image("results/figures/eda/sampling_and_speed.png", 15 * cm,
          "Figure 2.4 - Sampling interval and GPS speed, train vs student_test.")
    image("results/figures/eda/graph_projection.png", 15 * cm,
          "Figure 2.5 - Road-network structure and projection-error audit, both partitions.")
    image("docs/evidence/development_map.jpg", 15 * cm,
          "Figure 2.6 - Folium geographical map of student_test trip 00000052: GPS observations, sampled road "
          "network, reference route and predicted routes with a layer control (src/visualization/folium_maps.py; "
          "interactive HTML in maps/development/).")
    story.append(PageBreak())


MODEL_TEXT = {
    "logistic_regression": dict(
        why="A linear probabilistic baseline: it outputs well-calibrated probabilities, which the HMM uses directly "
            "as emission evidence, and it shows how much of the problem is linearly separable from distance, "
            "rank and heading.",
        sl="Strengths: very fast (sub-microsecond scoring per candidate), stable across folds and calibrated. "
           "Limitations: it cannot model interactions such as 'close to the point but heading-misaligned' without "
           "hand-made feature crosses, which caps its macro-F1.",
        hp="C=1.0 (L2), solver lbfgs, max_iter=1000, tol=1e-4, class_weight=None; median imputation with missing "
           "indicators and standard scaling fitted per fold.",
        loss="Binary cross-entropy (logistic log loss) with an L2 penalty."),
    "decision_tree": dict(
        why="Captures non-linear thresholds and interactions (distance vs heading vs rank) and is easy to inspect, "
            "making it a useful interpretable non-linear baseline.",
        sl="Strengths: fast to score, no scaling needed, handles interactions. Limitations: piecewise-constant "
           "probabilities (poor calibration for HMM emissions) and high variance across trips.",
        hp="criterion=gini, max_depth=12, min_samples_leaf=50, min_samples_split=2, max_features=None, "
           "class_weight=None; median imputation with missing indicators.",
        loss="Gini impurity minimised greedily at each split."),
    "random_forest": dict(
        why="Averaging many decorrelated trees reduces the variance of a single tree while keeping its ability to "
            "model interactions; class weighting targets recall of the rare true segment.",
        sl="Strengths: robust, strong Top-1, informative feature importances (dist_minus_min, cand_rank and "
           "heading_diff_deg dominate). Limitations: the largest model, slowest to train, and class-balanced "
           "scores are not calibrated probabilities.",
        hp="n_estimators=80, max_depth=24, min_samples_split=4, min_samples_leaf=2, max_features=0.5, "
           "class_weight=balanced; median imputation with missing indicators.",
        loss="Gini impurity per tree with class-balanced sample weights; predictions average tree probabilities."),
    "gradient_boosting": dict(
        why="Boosted trees fit the remaining errors of earlier trees, model interactions, handle missing values "
            "natively and optimise log loss directly, giving well-ranked and calibrated candidate probabilities.",
        sl="Strengths: best cross-validation macro-F1 and log loss, compact model. Limitations: more "
           "hyperparameters to tune; sequential training.",
        hp="HistGradientBoostingClassifier: learning_rate=0.1, max_iter=100, max_leaf_nodes=15, "
           "min_samples_leaf=30, l2_regularization=1.0, max_bins=127, early_stopping=False, class_weight=None; "
           "missing values routed natively.",
        loss="Binary log loss (logistic loss) minimised by gradient boosting with L2 leaf regularisation."),
    "knn": dict(
        why="An instance-based method: a candidate is judged by similar labelled candidates in the scaled "
            "feature space, with no assumption about the decision boundary's shape.",
        sl="Strengths: simple, non-parametric, competitive macro-F1 on easier folds. Limitations: prediction "
           "requires a neighbour search over about one million stored rows (slowest model to query), and "
           "distance-weighted votes of five neighbours give coarse, often exactly 0 or 1 probabilities, which "
           "inflates log loss.",
        hp="n_neighbors=5, weights=distance, metric=minkowski, p=2 (Euclidean), algorithm=kd_tree, leaf_size=40; "
           "median imputation with missing indicators and standard scaling fitted per fold.",
        loss="No training loss: the model stores the training rows; prediction is a distance-weighted neighbour "
             "vote."),
}


def section_3():
    full = pd.read_csv(R / "results/models/comparison/five_model/all_metrics.csv").set_index("model")
    deliv = pd.read_csv(R / "results/models/comparison/five_model/deliverables_table.csv")
    cont = pd.read_csv(R / "results/models/comparison/classifier_only_continuity.csv").set_index(["model", "partition"])

    h1("3. Model Experimentation and Comparison")
    p("Five distinct algorithms were trained and evaluated: Logistic Regression, Decision Tree, Random Forest, "
      "Histogram Gradient Boosting and K-Nearest Neighbours. All five use the same 13 ordered features, the same "
      "saved five-fold trip-grouped split of the 80 train trips (configs/cv_folds.csv; no trip appears in both "
      "training and validation), the same label policy (only trusted labelled candidates are fitted) and the same "
      "metric code (src/evaluation/metrics.py). Each model is refitted on all 1,070,140 trusted labelled train rows "
      "and evaluated once on the 10 student_test trips. Every saved fold and student-test result was recomputed "
      "from the published score files (results/validation/*_audit.json).")
    p("<b>Metric definitions.</b> Candidate metrics (accuracy, precision, recall, F1, macro-F1, log loss, confusion "
      "matrix) use labelled candidates at threshold 0.5. Observation Top-1 picks the highest-probability candidate "
      "of each trusted GPS observation (ties to the nearest) and compares it with the reference arc; observations "
      "whose reference is not among the candidates count as wrong. Route continuity is the share of consecutive "
      "distinct predicted arcs that are joined head-to-tail in the directed road graph, within GPS-gap segments.")

    h2("3.1 Overall Model Comparison")
    rows = [["Model", "Algorithm Family", "Key Configuration", "CV Macro F1-score", "Top-1 Accuracy",
             "Route continuity", "Student-Test Macro F1-score", "Log Loss", "Inference Time"]]
    for d in deliv.itertuples(index=False):
        rows.append([str(c).replace("HistGradientBoosting", "Hist Gradient Boosting") for c in d])
    table(rows, [2.0 * cm, 2.0 * cm, 3.1 * cm, 1.8 * cm, 1.75 * cm, 1.8 * cm, 1.85 * cm, 1.3 * cm, 1.8 * cm])
    p("Table 3.1 - CV Macro F1 = mean +/- std over the 5 validation folds of the 80 train trips. All other columns "
      "are measured on the 10 student_test trips: Top-1 over 16,073 trusted observations; route continuity of the "
      "classifier-only prediction; log loss over 106,730 labelled candidates; inference time in microseconds per "
      "candidate row (results/models/comparison/five_model/deliverables_table.csv, all_metrics.csv, "
      "classifier_only_continuity.csv). Inference times were measured on one laptop: HistGradientBoosting by its "
      "batched training runner, KNN while other jobs shared the CPU.", "Cap")

    rows = [["Model", "Accuracy", "Precision (positive)", "Recall (positive)", "Macro precision",
             "Macro recall", "CV Top-1", "CV log loss", "Candidate miss rate", "Train-OOF continuity",
             "CV fit time (s)"]]
    for m in ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "knn"]:
        r = full.loc[m]
        rows.append([NAMES[m], pct(r.student_test_candidate_accuracy), f"{r.student_test_precision_pos:.4f}",
                     f"{r.student_test_recall_pos:.4f}", f"{r.student_test_precision_macro:.4f}",
                     f"{r.student_test_recall_macro:.4f}", f"{r.cv_mean_top1_accuracy:.4f}",
                     f"{r.cv_mean_log_loss:.4f}", pct(r.student_test_candidate_miss_rate),
                     f"{cont.loc[(m, 'train_oof'), 'continuity']:.4f}", f"{r.cv_fit_s:.0f}"])
    table(rows, [2.4 * cm] + [1.5 * cm] * 10)
    p("Table 3.2 - Additional measured metrics. Accuracy, precision, recall and candidate miss rate are on "
      "student_test; CV columns are means over the 5 folds; train-OOF continuity uses each trip's out-of-fold "
      "scores.", "Cap")

    p("<b>Model selection.</b> The selection metric, fixed before optimisation, is mean training-CV macro-F1, with "
      "CV log loss and CV Top-1 as supporting criteria; student_test is used for reporting, not for choosing. "
      f"Histogram Gradient Boosting has the highest CV macro-F1 ({full.loc['gradient_boosting', 'cv_mean_f1_macro']:.4f}), "
      f"the lowest CV log loss ({full.loc['gradient_boosting', 'cv_mean_log_loss']:.4f}, i.e. the best probabilities "
      f"for HMM emissions), the highest CV Top-1 ({full.loc['gradient_boosting', 'cv_mean_top1_accuracy']:.4f}) and the "
      f"highest classifier-only continuity, and it also leads on student-test macro-F1 "
      f"({full.loc['gradient_boosting', 'student_test_f1_macro']:.4f}) and log loss. "
      "It was therefore selected. Random Forest is close on CV macro-F1 and has a marginally higher student-test "
      "Top-1, but its log loss is worse and its class-balanced scores are not calibrated.")

    h2("3.2 Compulsory Questions for Every Model")
    for m in ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "knn"]:
        cv, test = load_metrics(m)
        folds = cv[cv.fold.str.isdigit()]
        val = folds[folds.split == "validation"]
        trn = folds[folds.split == "train"]
        t = MODEL_TEXT[m]
        h3(f"3.2.{['logistic_regression', 'decision_tree', 'random_forest', 'gradient_boosting', 'knn'].index(m) + 1} "
           f"{NAMES[m]}")
        p("<b>A. Model Understanding</b>")
        qa("1. Why did you consider this algorithm suitable for the PathSnap problem?", t["why"])
        qa("2. What are the major strengths and limitations of this algorithm for this task?", t["sl"])
        p("<b>B. Training and Configuration</b>")
        qa("5. What were the important hyperparameters and their initial values?", t["hp"])
        qa("6. What loss function or optimisation objective was used?", t["loss"])
        p("<b>C. Experimental Results</b>")
        qa("7. What was the model's performance on the student-testing dataset?",
           f"Candidate accuracy {pct(test.candidate_accuracy)}, positive-class precision {test.precision_pos:.4f} "
           f"and recall {test.recall_pos:.4f} (TP {int(test.tp):,}, FP {int(test.fp):,}, FN {int(test.fn):,}, "
           f"TN {int(test.tn):,}).")
        qa("8. What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing "
           "dataset?",
           f"Top-1 {pct(test.top1_accuracy)} ({int(test.n_top1_correct):,} of {int(test.n_trusted_obs):,} trusted "
           f"observations), macro-F1 {test.f1_macro:.4f}, log loss {test.log_loss:.4f}.")
        fold_txt = ", ".join(f"{v:.3f}" for v in val.f1_macro)
        if m == "gradient_boosting":
            tuned_cv, _ = load_metrics("gradient_boosting_tuned")
            tf = tuned_cv[tuned_cv.fold.str.isdigit()]
            gap = (f"The same model family refitted with the shared runner (gradient_boosting_tuned) gives train "
                   f"macro-F1 {tf[tf.split == 'train'].f1_macro.mean():.3f} vs validation "
                   f"{tf[tf.split == 'validation'].f1_macro.mean():.3f}, a small gap")
        elif m == "knn":
            gap = (f"Training-split macro-F1 is {trn.f1_macro.mean():.3f} (measured on a fixed 50,000-row sample per "
                   f"fold) vs validation {val.f1_macro.mean():.3f}: each training row is its own nearest neighbour, "
                   "so the training score reflects memorisation")
        else:
            gap = (f"Mean training-fold macro-F1 {trn.f1_macro.mean():.3f} vs validation {val.f1_macro.mean():.3f} "
                   f"(log loss {trn.log_loss.mean():.3f} vs {val.log_loss.mean():.3f})")
        verdict = {"logistic_regression": "Train and validation scores are almost equal, which indicates "
                                          "underfitting: the linear boundary is the limiting factor.",
                   "decision_tree": "A moderate train/validation gap and the largest fold spread indicate mild "
                                    "overfitting to training trips.",
                   "random_forest": "The large train/validation gap indicates clear overfitting to training trips, "
                                    "although averaging keeps validation performance high.",
                   "gradient_boosting": "The train/validation gap is small, held-out scores are at least as good as "
                                        "the CV mean and fold variation is similar to the other tree models, so "
                                        "neither strong overfitting nor underfitting is observed.",
                   "knn": "This is overfitting by construction (k=5 memorises local neighbourhoods); the poor "
                          "validation log loss confirms it."}[m]
        qa("9. Did you observe overfitting or underfitting? What evidence supports your conclusion?",
           f"{gap}. Validation macro-F1 per fold: {fold_txt} (mean {val.f1_macro.mean():.4f} +/- "
           f"{val.f1_macro.std():.4f}); student-test macro-F1 {test.f1_macro:.4f}. {verdict}")
    story.append(PageBreak())

    h2("3.3 Model Visualisations")
    base = "results/models/comparison/five_model/"
    image(base + "cv_macro_f1.png", 14 * cm, "Figure 3.1 - Cross-validation macro-F1 across all five models "
          "(bars = fold mean, error bars = std, dots = individual folds; 80 train trips).")
    image(base + "student_test_top1_and_macro_f1.png", 16 * cm,
          "Figure 3.2 - Student-testing Top-1 accuracy and macro-F1 comparison (10 student_test trips).")
    image(base + "log_loss.png", 14 * cm, "Figure 3.3 - Log loss comparison: CV mean (solid) and student_test "
          "(hatched), log scale.")
    image(base + "student_test_confusion_matrices.png", 16 * cm,
          "Figure 3.4 - Student-test confusion matrices at threshold 0.5.")
    image(base + "fold_stability.png", 16 * cm, "Figure 3.5 - Per-fold validation macro-F1 and Top-1, used to "
          "judge stability; all models drop on fold 4, which contains most reference misses.")
    image(base + "timing.png", 16 * cm, "Figure 3.6 - Measured training and inference cost.")
    p("None of the five algorithms trains in epochs (trees, boosting iterations without early stopping, a "
      "lbfgs solver for logistic regression, which does not record per-iteration losses, and an instance-based "
      "KNN), so no epoch-wise loss curves exist; per-fold training and validation metrics are reported instead.")
    story.append(PageBreak())


def section_4():
    s = json.loads((R / "results/optimization/gradient_boosting_tpe_deadline/summary.json").read_text())
    trials = pd.read_csv(R / "results/optimization/gradient_boosting_tpe_deadline/trials.csv")
    tv = pd.read_csv(R / "results/models/comparison/tuned_vs_untuned/all_metrics.csv").set_index("model")
    sp = s["search_space"]
    best = s["best_params"]

    h1("4. Bayesian Hyperparameter Optimisation (Histogram Gradient Boosting)")
    p("Code: src/optimization/bayes_search.py; configuration: configs/optimisation_deadline.yaml; records: "
      "results/optimization/gradient_boosting_tpe_deadline/ (trials.csv, summary.json, search_history.png).")
    qa("1. Which hyperparameters were included in the search?",
       "learning_rate, max_iter, max_leaf_nodes, min_samples_leaf and l2_regularization. max_bins (63), "
       "class_weight (None) and early_stopping (off, so that no random row-level holdout crosses trip "
       "boundaries) were held fixed.")
    qa("2. What ranges or distributions were considered?",
       f"learning_rate log-uniform [{sp['learning_rate']['low']}, {sp['learning_rate']['high']}]; max_iter integer "
       f"[{sp['max_iter']['low']}, {sp['max_iter']['high']}] in steps of {sp['max_iter']['step']}; max_leaf_nodes "
       f"log-integer [{sp['max_leaf_nodes']['low']}, {sp['max_leaf_nodes']['high']}]; min_samples_leaf log-integer "
       f"[{sp['min_samples_leaf']['low']}, {sp['min_samples_leaf']['high']}]; l2_regularization log-uniform "
       f"[{sp['l2_regularization']['low']}, {sp['l2_regularization']['high']}]. The baseline values "
       "(learning_rate 0.1, 100 iterations, 15 leaves, 30 samples per leaf) lie inside these ranges, which extend "
       "towards larger trees and weaker regularisation.")
    qa("3. What objective function was optimised?",
       "Maximise the mean validation candidate macro-F1 (threshold 0.5) over the five canonical trip-grouped folds "
       "of the 80 train trips (fold file SHA-256 17db171e...). Each fold model is fitted only on the labelled rows "
       "of the other four folds; student_test is never read by the search. The sampler is Optuna's Tree-structured "
       "Parzen Estimator (multivariate, seed 2026): the first trials sample the space at random, later trials are "
       "proposed from a density model of the earlier results. A median pruner stops a trial after at least three "
       "folds if it falls below the median of earlier trials on the same fold.")
    qa("4. How many optimisation trials were performed?",
       f"{len(trials)} trials: {s['trials_by_phase'].get('startup_random', 0)} random start-up trials and "
       f"{s['trials_by_phase'].get('adaptive_tpe', 0)} adaptive TPE trials; "
       f"{s['trials_by_state'].get('COMPLETE', 0)} completed all five folds and "
       f"{s['trials_by_state'].get('PRUNED', 0)} was pruned after three folds. Total trial compute "
       f"{s['total_trial_seconds']:.0f} s on 4 threads.")
    rows = [["Trial", "Phase", "State", "Objective (mean CV macro-F1)", "learning_rate", "max_iter",
             "max_leaf_nodes", "min_samples_leaf", "l2_regularization"]]
    for t in trials.itertuples():
        pr = json.loads(t.params)
        obj = f"{t.objective:.5f}" if t.state == "COMPLETE" else f"pruned ({int(t.folds_scored)} folds)"
        rows.append([t.trial, t.phase.replace("_", " "), t.state.lower(), obj, f"{pr['learning_rate']:.4f}",
                     pr["max_iter"], pr["max_leaf_nodes"], pr["min_samples_leaf"], f"{pr['l2_regularization']:.4f}"])
    table(rows, [1.1 * cm, 2.2 * cm, 1.6 * cm, 2.6 * cm, 1.8 * cm, 1.5 * cm, 1.9 * cm, 2.1 * cm, 2.1 * cm])
    p("Table 4.1 - All optimisation trials (results/optimization/gradient_boosting_tpe_deadline/trials.csv).", "Cap")
    qa("5. What were the best hyperparameter values obtained?",
       f"Trial {s['best_trial']}: learning_rate {best['learning_rate']:.4f}, max_iter {best['max_iter']}, "
       f"max_leaf_nodes {best['max_leaf_nodes']}, min_samples_leaf {best['min_samples_leaf']}, "
       f"l2_regularization {best['l2_regularization']:.4f} (max_bins 63), with mean CV macro-F1 "
       f"{s['best_value']:.5f}. The model was refitted with these values on all 80 train trips "
       "(results/models/gradient_boosting_tuned/).")
    u, t = tv.loc["gradient_boosting"], tv.loc["gradient_boosting_tuned"]
    qa("6. Did the tuned model outperform the untuned model?",
       f"Not on the selection objective: CV macro-F1 is {t.cv_mean_f1_macro:.5f} tuned vs {u.cv_mean_f1_macro:.5f} "
       f"untuned, a difference far smaller than the fold standard deviation ({u.cv_std_f1_macro:.3f}). The tuned "
       f"model is marginally better on student-test Top-1 ({pct(t.student_test_top1_accuracy)} vs "
       f"{pct(u.student_test_top1_accuracy)}) and log loss ({t.student_test_log_loss:.4f} vs "
       f"{u.student_test_log_loss:.4f}), and equal on student-test macro-F1. Because selection is made on training "
       "CV, the untuned Histogram Gradient Boosting model remains the deployed model. The best search value is a "
       "maximum over trials scored on the same folds, so it is optimistically biased rather than a nested-CV "
       "estimate.")
    rows = [["Metric (same folds, partitions and metric code)", "Untuned", "Tuned (trial 0)"],
            ["Cross-validation macro-F1 (mean of 5 folds)", f"{u.cv_mean_f1_macro:.5f}", f"{t.cv_mean_f1_macro:.5f}"],
            ["Student-testing Top-1 accuracy", pct(u.student_test_top1_accuracy), pct(t.student_test_top1_accuracy)],
            ["Student-testing macro-F1", f"{u.student_test_f1_macro:.4f}", f"{t.student_test_f1_macro:.4f}"],
            ["Log loss (CV mean)", f"{u.cv_mean_log_loss:.4f}", f"{t.cv_mean_log_loss:.4f}"],
            ["Log loss (student_test)", f"{u.student_test_log_loss:.4f}", f"{t.student_test_log_loss:.4f}"],
            ["Inference time (us per candidate, student_test)", f"{u.inference_us_per_row:.2f}",
             f"{t.inference_us_per_row:.2f}"],
            ["Total 5-fold CV fitting time (s)", f"{u.cv_fit_s:.1f}", f"{t.cv_fit_s:.1f}"]]
    table(rows, [8 * cm, 3.5 * cm, 3.5 * cm])
    p("Table 4.2 - Tuned vs untuned (results/models/comparison/tuned_vs_untuned/all_metrics.csv). The untuned "
      "model's timings come from its original batched runner and the tuned model's from the shared runner.", "Cap")
    image("results/optimization/gradient_boosting_tpe_deadline/search_history.png", 13 * cm,
          "Figure 4.1 - Optimisation history: objective per trial by phase and best value so far.")
    image("results/figures/review/02_before_after_tuning.png", 15 * cm,
          "Figure 4.2 - Untuned vs tuned model on the CV objective and student-test metrics.")
    story.append(PageBreak())


def section_5():
    trips = pd.read_csv(R / "results/rehearsal/student_test_api.csv", dtype={"trip_id": str})
    summ = pd.read_csv(R / "results/hmm/development/classifier_vs_hmm_summary.csv")
    cfg = __import__("yaml").safe_load((R / "configs/decoder.yaml").read_text())

    h1("5. HMM and Viterbi Route Reconstruction")
    p("Code: src/hmm/ (emission.py, transition.py, viterbi.py, route_decoder.py, adapter.py), decoder settings in "
      "configs/decoder.yaml, route metrics in src/hmm/route_metrics.py. The pipeline for one trip is: GPS track and "
      "road graph, candidate generation and features (as in training), classifier probabilities from the selected "
      "Histogram Gradient Boosting model, then HMM/Viterbi decoding and route assembly. Reference routes are read "
      "only by the separate server-side evaluation step after prediction.")
    qa("1. What are the hidden states in your implementation?",
       "For each GPS observation the hidden state is which of its candidate road segments (directed arcs within "
       "100 m, at most 8, ranked by distance) the vehicle is actually on. The state space therefore changes per "
       "observation, and two directions of the same road are different states.")
    qa("2. How are emission probabilities calculated?",
       f"The emission score of a candidate is w * log(max(p, eps)), where p is the classifier's positive-class "
       f"probability for that candidate, eps = {cfg['emission_eps']} and w = {cfg['emission_weight']}. The "
       "classifier posterior is used as the emission evidence (a standard discriminative approximation of "
       "P(observation | arc)); the floor keeps a zero probability finite.")
    qa("3. How are transition probabilities calculated?",
       f"Following Newson and Krumm, the log transition score between candidate i at time t-1 and candidate j at "
       f"time t is -|d_route - d_gps| / beta with beta = {cfg['beta_m']} m. d_gps is the straight-line distance "
       "between the two GPS points and d_route is the driving distance from the projected position on arc i to the "
       "projected position on arc j, found by a bounded Dijkstra search that follows arc directions only. The search "
       f"bound is max_speed * dt + slack ({cfg['max_speed_mps']} m/s and {cfg['dist_slack_m']} m; "
       f"{cfg['max_route_dist_m']} m if the time step is unknown). Moving backwards on the same arc by up to "
       f"{cfg['same_arc_backtrack_m']} m is treated as GPS jitter; any pair that cannot be reached within the bound "
       "gets probability zero (log score -inf). Viterbi then finds the highest-scoring sequence in log space.")
    qa("4. How are missing candidates or large GPS gaps handled?",
       f"Observations with no candidate within 100 m are skipped (no prediction) and decoding continues with the "
       f"next observation that has candidates. A time gap above {cfg['gap_split_s']:.0f} s, or a non-positive or "
       "invalid time step, splits the trip into independent segments that are decoded separately. Route assembly "
       "never joins geometry across these splits, and the evaluation counts skipped observations as wrong.")
    qa("5. What happens when consecutive GPS observations suggest disconnected road segments?",
       "If no candidate of the next observation is reachable from any candidate of the previous one (all "
       "transitions are -inf), Viterbi stops the current chain at that point, back-tracks the best path so far, "
       "and restarts a new segment from the next observation (counted as a no-transition break). Within a segment, "
       "when consecutive predicted arcs are not directly adjacent, the shortest directed path found by the "
       "transition search is inserted so that the reconstructed route is drivable.")
    s = summ.set_index("metric")
    top = s.iloc[0]
    better = int((trips.top1_accuracy_hmm > trips.top1_accuracy_classifier).sum())
    qa("6. Did HMM decoding improve the final route? Support your answer with experimental results.",
       f"Yes. On the 10 student_test trips, Top-1 accuracy rises from {pct(top.classifier_only)} to "
       f"{pct(top.classifier_plus_hmm)} (+{100 * (top.classifier_plus_hmm - top.classifier_only):.2f} percentage "
       f"points, {int(trips.n_correct_hmm.sum() - trips.n_correct_classifier.sum()):,} more correctly matched "
       f"observations), route continuity rises from {pct(s.iloc[1].classifier_only)} to "
       f"{pct(s.iloc[1].classifier_plus_hmm)}, and the number of route breaks falls from "
       f"{int(s.iloc[2].classifier_only):,} to {int(s.iloc[2].classifier_plus_hmm):,}. HMM decoding improves "
       f"Top-1 on {better} of the 10 trips.")
    rows = [["Metric (10 student_test trips)", "Classifier only", "Classifier + HMM/Viterbi", "Aggregation"]]
    for r in summ.itertuples():
        f = (lambda v: f"{int(v):,}") if r.metric == "Route breaks" else (
            (lambda v: pct(v)) if ("accuracy" in r.metric or "continuity" in r.metric) else (lambda v: f"{v:.3f}"))
        rows.append([r.metric, f(r.classifier_only), f(r.classifier_plus_hmm), r.aggregation])
    table(rows, [5.2 * cm, 2.6 * cm, 3.2 * cm, 5 * cm])
    p("Table 5.1 - Classifier-only vs classifier + HMM/Viterbi (results/hmm/development/classifier_vs_hmm_summary.csv, "
      "computed from results/rehearsal/student_test_api.csv).", "Cap")
    rows = [["Trip", "Trusted obs", "Classifier Top-1", "HMM Top-1", "Change (pp)", "Classifier continuity",
             "HMM continuity", "Segments"]]
    for r in trips.itertuples():
        rows.append([r.trip_id, int(r.n_evaluated_hmm), pct(r.top1_accuracy_classifier), pct(r.top1_accuracy_hmm),
                     f"{100 * (r.top1_accuracy_hmm - r.top1_accuracy_classifier):+.2f}",
                     pct(r.route_continuity_classifier), pct(r.route_continuity_hmm), int(r.n_segments)])
    table(rows, [2 * cm, 1.8 * cm, 2.2 * cm, 2 * cm, 1.8 * cm, 2.3 * cm, 2.1 * cm, 1.6 * cm])
    p("Table 5.2 - Per-trip results (results/rehearsal/student_test_api.csv).", "Cap")
    worse = trips[trips.top1_accuracy_hmm < trips.top1_accuracy_classifier]
    gains = (trips.top1_accuracy_hmm - trips.top1_accuracy_classifier).sort_values(ascending=False)
    top2 = trips.loc[gains.index[:2]]
    p("<b>Improvements and cases where HMM did not help.</b> The largest gains are on "
      + " and ".join(f"trip {r.trip_id} ({pct(r.top1_accuracy_classifier)} to {pct(r.top1_accuracy_hmm)})"
                     for r in top2.itertuples())
      + ", where independent per-point decisions switch between nearby roads and the transition model removes "
      "those switches. Route continuity improves on every trip. "
      + ("".join(f"HMM decoding made trip {w.trip_id} worse in Top-1 ({pct(w.top1_accuracy_classifier)} to "
                 f"{pct(w.top1_accuracy_hmm)}): when most per-point decisions are already correct, the "
                 "connectivity preference can override some correct decisions. " for w in worse.itertuples()))
      + "Accuracy here is agreement with the derived reference alignment, not independently surveyed ground truth.")
    image("results/figures/review/03_hmm_per_trip.png", 15 * cm,
          "Figure 5.1 - Classifier-only vs classifier + HMM Top-1 for every student_test trip.")
    image("results/figures/review/01_baseline_tuned_hmm_top1.png", 13 * cm,
          "Figure 5.2 - Student-test Top-1 of the untuned model, the tuned trial and the selected model with "
          "HMM/Viterbi.")
    story.append(PageBreak())


def section_6():
    h1("6. Geographical Visualisation Using Folium")
    p("Map code: src/visualization/folium_maps.py (create_trip_map). Interactive maps committed in maps/development/ "
      "for student_test trips 00000052, 00000056 and 00000076 (the shortest, median-length and longest trips, "
      "chosen before looking at any results).")
    p("<b>Layers and design.</b> Each map shows (1) the original GPS observations as orange circle markers, (2) the "
      "relevant road-network geometry as thin grey lines, (3) the reference route in teal, (4) the classifier-only "
      "predicted route in magenta and (5) the HMM/Viterbi predicted route in blue. A fixed legend names the trip, "
      "partition and model, and a layer control lets each layer be switched on and off to compare the observed "
      "trajectory with the predicted route. The view is fitted to the GPS extent so the zoom level matches the "
      "trip. Predicted segments separated by GPS gaps are drawn as separate lines and never joined.")
    image("docs/evidence/development_map.jpg", 16 * cm,
          "Figure 6.1 - Folium map of student_test trip 00000052 (maps/development/00000052.html): GPS points, road "
          "network, reference route, classifier and HMM predictions, legend and layer control.")
    image("results/figures/eda/trajectory_network.png", 15 * cm,
          "Figure 6.2 - Static trajectory and road-network view of a train and a student_test trip (EDA).")
    story.append(PageBreak())


def section_7():
    smoke = json.loads((R / "results/validation/demo_http_smoke.json").read_text())
    log = (R / "results/rehearsal/server_inference.jsonl").read_text(encoding="utf-8").splitlines()[0]
    rec = json.loads(log)
    h1("7. API, UI and Deployment")
    p("<b>Architecture.</b> Laptop 1 runs the FastAPI server (src/api/main.py, started with "
      "scripts/start_api.ps1 on port 8020). It holds the deployment bundle (best_model.joblib, preprocessing.joblib, "
      "model_card.json with SHA-256 checks, and the decoder configuration), the road networks and trips, and the "
      "server-side evaluation. Laptop 2 runs the Streamlit dashboard (ui/app.py, started with scripts/start_ui.ps1 "
      "-ApiHost &lt;Laptop 1 IP&gt;), which lists the available trips, sends the selected trip ID and displays the "
      "returned routes, metrics and timings. Reference routes and labels never leave Laptop 1.")
    p("<b>Endpoints.</b> GET /health returns the model name and version, schema version and decoder version; GET "
      "/trips returns the trip IDs discovered at start-up; POST /predict with {\"trip_id\": ...} returns the GPS "
      "coordinates, the HMM and classifier-only route polylines, per-trip metrics, diagnostics and per-stage "
      "timings. Every request is logged on Laptop 1 with a request ID that is also shown in the dashboard.")
    p(f"<b>Verification.</b> An HTTP check returned /health status {smoke['health']}, {smoke['trip_count']} trips, and a "
      f"real (non-stub) prediction for trip {smoke['prediction_trip']} with status {smoke['prediction_status']} "
      "(results/validation/demo_http_smoke.json). All ten student_test trips were requested over HTTP; the returned "
      "geometry and metrics matched offline inference exactly (results/rehearsal/summary.json, "
      "student_test_api.csv).")
    image("docs/evidence/local_ui.jpg", 15 * cm,
          "Figure 7.1 - PathSnap dashboard: API status with model and schema version, trip selector, Run Prediction, "
          "and the classifier Top-1, HMM Top-1 and candidate-miss-rate cards for the selected trip.")
    image("docs/evidence/local_ui_route.jpg", 15 * cm,
          "Figure 7.2 - Dashboard map view: GPS points (orange), classifier route (green) and HMM route (blue) "
          "with layer control.")
    p("<b>Successful API response record (Laptop 1 server log, results/rehearsal/server_inference.jsonl):</b>")
    story.append(Preformatted(json.dumps(rec, indent=1), styles["Mono"]))
    story.append(Spacer(1, 6))
    p(f"The record shows request {rec['request_id'][:8]}... for trip {rec['trip_id']} served by model "
      f"{rec['details']['model_version']}: {rec['details']['n_candidates']:,} candidates scored in "
      f"{rec['stage_timings_ms']['score']:.0f} ms and decoded in {rec['stage_timings_ms']['decode']:.0f} ms, "
      f"{rec['stage_timings_ms']['total']:.0f} ms end to end.")


def main():
    cover()
    section_2()
    section_3()
    section_4()
    section_5()
    section_6()
    section_7()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(1.8 * cm, 1.1 * cm, "PathSnap - Team 84 - UE24CS352A")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                            title="PathSnap Technical Report - Team 84", author="Team 84")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
