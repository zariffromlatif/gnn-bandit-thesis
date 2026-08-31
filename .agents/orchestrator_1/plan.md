# Orchestration Plan: GNN-Bandit Research Review & Q1 Publication Readiness

## Overview
Comprehensive research review, theoretical/methodological integrity audit, experimental statistical significance verification, and publication blueprint formulation for GNN-Bandit targeting Q1 journals.

## Tracks & Milestones
- **Milestone 1 (R1 - Empirical Suite & Statistical Significance Audit)**:
  - Survey all result files (`experiments/results/`, `experiments/results-v2-lambda-*/`, `experiments/results-cfr/`, etc.).
  - Run paired t-tests and Wilcoxon signed-rank tests across 5 seeds comparing GNN-Bandit against 9+ baselines across OBD-All, OBD-Men, OBD-Women, and Criteo.
  - Assess sensitivity, cold-start, lambda ablation, and anomaly diagnosis (e.g. CQL on Criteo).
- **Milestone 2 (R2 - Codebase & Methodological Integrity Review)**:
  - Audit `src/graph/lightgcn.py`, `src/graph/tgn.py`, `src/causal/cate_estimator.py`, `src/agent/bcq.py`, `src/agent/bcq_dynamic.py`, `src/ope/estimators.py`, `src/utils/data_loader.py`.
  - Check for theoretical correctness, absence of data leakage/lookahead bias, offline bandit / causal uplift rigor.
  - Forensic verification of authenticity and code integrity.
- **Milestone 3 (R3 - Q1 Journal Gap Analysis & Publication Blueprint)**:
  - Benchmark against KBS, ESWA, ACM TOIS/TORS, IEEE TKDE reviewer standards.
  - Formulate mathematical notation, manuscript section drafts, LaTeX tables, figure suggestions, and gap matrix.
- **Synthesis & Final Reporting**:
  - Consolidate all audit reports, significance tables, gap analysis, and manuscript blueprint into comprehensive handoff and parent report.
