# Completion matrix: preparation evidence, not final acceptance

Page numbers refer to the supplied official PDFs, not the old team plan.

| Requirement and source | Evidence / current status |
|---|---|
| Directed HMM/Viterbi, all observation identities, missing/gap/impossible transitions (Models and Training Guidelines pp.4-5) | Implemented in src/hmm; brute-force, direction, gap, shuffled key and numerical tests. Reuses reviewed route branch. |
| Pointwise, continuity, sequence, miss-rate and measured comparison (Models pp.3-5; Deliverables pp.5-6) | Shared helpers, correct count denominators, ten-trip baseline evidence under results/rehearsal. Final selected-model/OOF experiments pending A. |
| Five measured models and Bayesian optimisation (Assessment p.1; Models pp.1-4; Deliverables pp.3-5) | Person A dependency. RF/shared/KNN source merged into B branch from latest main; source alone is not completed experiments. |
| No evaluation leakage (Assessment p.3; Dataset Exploration p.3; Models p.6) | Actual evaluator ZIP sealed. Route-free prediction tested with altered, missing and forbidden reference reads. Server-only reference evaluation follows prediction. |
| API on Laptop 1; UI on Laptop 2; all ten trips (Main Guidelines p.2; Models pp.5-6) | Real baseline API/UI and dynamic registry implemented; all ten allowed stand-ins checked through HTTP against offline geometry/metrics. Physical two-laptop run pending. |
| Genuine receiving-console logs and screenshots (Models p.6; Deliverables p.6) | Real local structured logs, request IDs and UI screenshots retained. Actual second-machine console/UI screenshots pending physical rehearsal. |
| Geographic Folium layers and exported PDF view (Dataset pp.2-4; Deliverables pp.2,6) | Three actual student-test HTML maps, reusable generator, legend/layers/sampling notes, real captured map screenshot and working PDF. A's full EDA contribution pending. |
| Source/dependencies/reproduction/public GitHub (Deliverables pp.1-2) | Separate branch and draft PR #12, clean-environment tests, verified development release assets, documented startup and networking. Final frozen release pending final artifacts. |
| Final model/preprocessor/config export (Models p.5; Deliverables p.2) | Explicit development split bundle; final-only packaging and startup gates implemented. Final checksummed publication/download awaits A and tuning. |
| One consolidated PDF: data/EDA, every model, plots, Bayesian, HMM, Folium, deployment (Deliverables pp.2-7) | Editable working evidence draft and PDF; final combined report awaits A's measured sections, final decoder results, physical evidence and human review. |
| Accurate AI disclosure and defensible work (Assessment pp.2-3; Deliverables p.7) | Assistance disclosed; no fictional training, metrics, approvals or official evaluation. Team understanding/review still required. |

No acceptance row dependent on Person A, physical hardware or official authorization
is marked complete merely because implementation code exists.
