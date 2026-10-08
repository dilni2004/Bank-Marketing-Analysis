# Project Decision Log

This document serves as the canonical record of architectural and methodological decisions for the **Machine Learning Based Campaign Response Prediction for Bank Marketing** project, in adherence to Constitution Principle VI (*Document As You Build*).

---

## ADR-001: Validation Split Strategy: Selection of Stratified Random Holdout over Chronological Partitioning

- **Date**: 2026-10-08
- **Status**: Ratified
- **Authors**: Athukorala M. B. (Data Engineering), Siribaddana D. S. (Problem Framing & EDA), Nishshanka A. D. N. N. (Model Development), Nawarathna M. S. G. (Evaluation & Diagnostics)

### Context & Problem Statement

In the initial project proposal document ([2026-DS-12_Initial Submission.pdf](file:///Users/moni/Desktop/ML%20proj/docs/2026-DS-12_Initial%20Submission.pdf)), Section 1.2 and Section 7 proposed model validation that respects the chronological nature of telemarketing campaigns and temporal/economic context variables:

> *"A no-skill baseline and multiple alternative classifiers will be compared through validation designed to respect the dataset's chronological ordering."* (Section 7, p. xiii)  
> *"However, an apparently accurate model can still be unsuitable if it ... is evaluated using a split that leaks time-related patterns."* (Section 1.2, p. v)

However, the initial data preprocessing pipeline in [02_Data_Preprocessing_Pipeline.ipynb](file:///Users/moni/Desktop/ML%20proj/notebooks/02_Data_Preprocessing_Pipeline.ipynb) implemented a standard stratified random split:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
```

This divergence created a methodology mismatch between documented project scope and the implemented data engineering pipeline. A rigorous investigation was required to evaluate whether the project should adopt a strict chronological holdout (Option A) or retain the stratified random holdout (Option B), giving priority to the intended deployment scenario of predicting campaign response for future outbound calls.

---

### Options Considered

#### Option A – Temporal Holdout (Chronological 80/20 Partitioning)

- **Description**: Exploit the dataset's intrinsic row ordering (May 2008 to November 2010), allocating the first 80% of chronological records ($N=32,950$, May 2008 – mid 2010) to the development set and reserving the final 20% ($N=8,238$, mid 2010 – November 2010) as a temporal holdout set.
- **Theoretical Advantage**: Faithfully mimics out-of-time deployment where only historical data precedes the prediction point.
- **Empirical Findings & Fatal Flaws**:
  1. **Extreme Covariate Shift (Macroeconomic Regime Discontinuity)**:
     Due to the 2008 global financial crisis, macroeconomic indicators exhibit virtually non-overlapping distributions between the development and holdout periods:
     - `euribor3m`: Development range is $[1.299, 5.045]$ (mean: $4.269$). Holdout range is $[0.634, 1.299]$ (mean: $1.031$). The two sets share only a single boundary value ($1.299$) with zero common support across the distribution. Tree-based models (Random Forest, Gradient Boosting) cannot extrapolate into unobserved feature space and will route all holdout records to edge leaf nodes.
     - `emp.var.rate`: Development range is $[-1.8, 1.4]$ (mean: $+0.650$) vs Holdout range $[-3.4, -1.1]$ (mean: $-2.192$).
     - `nr.employed`: Development range is $[5099.1, 5228.1]$ (mean: $5195.14$) vs Holdout range $[4963.6, 5099.1]$ (mean: $5054.63$).
  2. **Severe Target Prior Collapse (Concept Drift)**:
     - Development set positive response rate: **$6.37\%$** ($2,100$ positives / $32,950$ records).
     - Holdout set positive response rate: **$30.83\%$** ($2,540$ positives / $8,238$ records).
     - A massive **$384\%$ increase** in the positive base rate occurs in the final campaign block. Over **$54.7\%$** of all positive subscribers in the entire dataset are concentrated in the holdout partition. Models trained and calibrated on a $6.4\%$ prior will severely underestimate response probabilities, breaking threshold calibration and business lift optimization.
  3. **Categorical Support Discontinuity**:
     The campaign month `month='sep'` ($570$ records) appears solely between row indices $37,887$ and $40,855$. Under a chronological split at index $32,950$, `month='sep'` has **0 occurrences in training data**, introducing out-of-vocabulary test anomalies.
  4. **Lack of True Calendar Timestamps**:
     The dataset records only `month` and `day_of_week` without calendar years or date timestamps, preventing principled time-series rolling windows or seasonality adjustments.

#### Option B – Stratified Random Holdout (80/20 Partitioning, `random_state=42`)

- **Description**: Partition the 41,188 records into an 80% development set ($N=32,950$) and a 20% holdout set ($N=8,238$) using stratified random sampling on target $y$.
- **Empirical Advantages**:
  1. **Perfect Target Balance Preservation**:
     - Full dataset positive rate: $11.2654\%$ ($4,640$ / $41,188$).
     - Development set positive rate: **$11.2656\%$** ($3,712$ / $32,950$).
     - Holdout set positive rate: **$11.2649\%$** ($928$ / $8,238$).
     - Absolute difference: $< 0.0001\%$, satisfying statistical stratification tolerances.
  2. **Uniform Covariate Representation**:
     Both partitions span the full range of macroeconomic variables (`euribor3m` $[0.634, 5.045]$ in both train and test) and all categorical combinations (all 10 months including September represented proportionally).
  3. **Unbiased Architecture Comparison**:
     Enables statistically sound 5-fold cross-validation on the development set and reliable generalization evaluation across candidate models (Logistic Regression, Random Forest, XGBoost/LightGBM, MLP) as mandated by the project rubric.
- **Trade-off Acknowledged**:
  Samples from the same calendar months and macroeconomic periods are distributed across development and holdout sets. However, in real banking operations, models are regularly retrained on rolling operational windows with current economic indicators rather than frozen across years of macroeconomic regime change.

---

### Decision Outcome

**Ratified Decision**: Retain **Option B (Stratified Random Holdout, 80/20 with `random_state=42`)** as the primary validation and preprocessing partitioning strategy for the project. Abandon Option A (Temporal Holdout) for pipeline generation and model benchmarking.

---

### Empirical Evidence & Justification

| Metric / Dimension | Option A: Temporal Split (80/20) | Option B: Stratified Random (80/20) | Advantage / Impact |
| :--- | :--- | :--- | :--- |
| **Development Size ($N_{\text{train}}$)** | 32,950 records | 32,950 records | Equal capacity |
| **Holdout Size ($N_{\text{test}}$)** | 8,238 records | 8,238 records | Equal capacity |
| **Train Positive Rate** | **$6.37\%$** ($2,100$ responders) | **$11.27\%$** ($3,712$ responders) | Option B preserves representative priors |
| **Test Positive Rate** | **$30.83\%$** ($2,540$ responders) | **$11.26\%$** ($928$ responders) | Option A exhibits extreme $384\%$ drift |
| **`euribor3m` Train Range** | $[1.299, 5.045]$ (mean: $4.269$) | $[0.634, 5.045]$ (mean: $3.618$) | Option B covers full business cycle |
| **`euribor3m` Test Range** | $[0.634, 1.299]$ (mean: $1.031$) | $[0.634, 5.000]$ (mean: $3.633$) | Option A has virtually zero overlap |
| **`nr.employed` Overlap** | Zero overlap ($[5099, 5228]$ vs $[4963, 5099]$) | Full overlap ($[4963, 5228]$) | Option A forces out-of-distribution failure |
| **`month='sep'` in Train** | **0 records** (100% in test set) | **456 records** (80.0% in train) | Option B guarantees category coverage |
| **Cross-Validation Validity** | Severely impaired (train folds at 3-6%) | High (stratified folds match holdout) | Option B supports reliable model selection |

**Core Justification**:
1. The primary business lens is **Campaign Response Prediction** (lead ranking for outbound calling capacity), not macroeconomic forecasting or regime-change adaptation.
2. Under Option A, evaluation results would reflect failure of static algorithms to extrapolate across extreme macroeconomic shocks rather than the models' ability to discriminate responder propensity from customer attributes.
3. Option B provides a fair, reproducible, and leakage-free benchmark where cross-validation performance directly reflects held-out test performance.

---

### Scope Reconciliation & Governance Alignment

1. **Reconciliation Statement**:
   This decision formally supersedes the tentative chronological validation proposal in Section 1.2 and Section 7 of [2026-DS-12_Initial Submission.pdf](file:///Users/moni/Desktop/ML%20proj/docs/2026-DS-12_Initial%20Submission.pdf). The final project deliverable, interim presentations, and reports will cite ADR-001 and provide this empirical justification for utilizing stratified holdout validation.
2. **Constitution Compliance**:
   - **Principle I (Reproducibility First)**: Split uses explicit deterministic seed `random_state=42`.
   - **Principle III (No Data Leakage)**: `train_test_split` is executed prior to any feature transformation. Preprocessing scalers and encoders are fitted strictly on `X_train` and applied to `X_test`.
   - **Principle IV (Evaluation Honesty)**: Target distributions ($11.2656\%$ train vs $11.2649\%$ test) are explicitly published; accuracy alone will not be used; ROC-AUC and PR-AUC will serve as headline metrics.
   - **Principle VI (Document As You Build)**: Documented prior to model training in `docs/decision_log.md`.
