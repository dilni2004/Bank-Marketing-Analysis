# Phase 3: Structural Model Trade-offs Synthesis

**Student Identity**: Nishshanka A. D. N. N. (IT24104095)  
**Module**: IT3091 Machine Learning  
**Project**: Campaign Response Prediction for Bank Marketing (Group ID: 2026-DS-12)  
**Deliverable Phase**: Phase 3 Model Development & Cross-Validation Benchmarking  
**Companion Artifacts**:
- Notebook: [`notebooks/03_Model_Development.ipynb`](file:///Users/moni/Desktop/ML%20proj/notebooks/03_Model_Development.ipynb)
- Serialized Candidate Bundle: [`models/candidate_models.joblib`](file:///Users/moni/Desktop/ML%20proj/models/candidate_models.joblib)
- Benchmark Metric Table: [`models/cv_benchmark_summary.csv`](file:///Users/moni/Desktop/ML%20proj/models/cv_benchmark_summary.csv)
- Feature Metadata: [`data/processed/feature_metadata.json`](file:///Users/moni/Desktop/ML%20proj/data/processed/feature_metadata.json)

---

## 1. Executive Summary & Analytical Scope

In Phase 3, we developed, trained, cross-validated, and systematically benchmarked a comprehensive portfolio of machine learning models across five distinct algorithmic families on the $N=32,940$ preprocessed training records ($71$ encoded features) of the UCI Bank Marketing dataset. 

In strict alignment with **ADR-001 (Stratified Random Holdout)**, all model selection and hyperparameter optimization decisions were conducted exclusively within a **Stratified 5-Fold Cross-Validation** protocol (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). The held-out test partition (`X_test_processed.csv`, `y_test.csv`, $N=8,236$) remained completely unread, unreferenced, and untouched, preserving genuine generalization diagnostics for Phase 4.

Furthermore, in strict adherence to the project constitution's **Data Leakage Prohibition**, the post-contact feature `duration` was completely excluded from all feature matrices, preventing artificially inflated performance metrics.

---

## 2. Cross-Validation Benchmark Performance Matrix

Because positive campaign subscription responses represent an acute minority class (~11.27%, $N_{pos}=3,711$ vs. $N_{neg}=29,229$), **Precision-Recall AUC (PR-AUC / Average Precision)** served as our primary optimization metric. Models were evaluated across a holistic, imbalance-aware evaluation suite:

| Rank | Candidate Model | Algorithmic Family | Val PR-AUC (Mean ± Std) | Val ROC-AUC (Mean ± Std) | Val Balanced Acc | Val F1-Minority | Val Brier Score | Train-Val Gap | Mean Fit Time |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **HistGradientBoosting (Tuned)** | **Gradient Boosting** | **0.4680 ± 0.0148** | **0.8013 ± 0.0090** | **0.7464** | **0.4688** | **0.1560** | **+0.0345** | **1.58s** |
| 2 | Random Forest (Tuned) | Bagging Ensemble | 0.4668 ± 0.0173 | 0.7989 ± 0.0109 | 0.7461 | 0.4764 | 0.1497 | +0.1131 | 0.74s |
| 3 | Random Forest (Balanced) | Bagging Ensemble | 0.4664 ± 0.0156 | 0.7992 ± 0.0112 | 0.7458 | 0.4796 | 0.1496 | +0.1328 | 0.46s |
| 4 | HistGradientBoosting (Balanced) | Gradient Boosting | 0.4662 ± 0.0137 | 0.7981 ± 0.0088 | 0.7469 | 0.4736 | 0.1510 | +0.0686 | 1.20s |
| 5 | MLP Classifier (Neural) | Neural Network | 0.4461 ± 0.0173 | 0.7881 ± 0.0142 | 0.6070 | 0.3379 | 0.0794 | +0.0453 | 1.21s |
| 6 | Logistic Regression (Balanced) | Linear Benchmark | 0.4454 ± 0.0128 | 0.7904 ± 0.0081 | 0.7402 | 0.4524 | 0.1644 | +0.0068 | 0.10s |
| 7 | Dummy Classifier (Prior) | No-Skill Baseline | 0.1127 ± 0.0001 | 0.5000 ± 0.0000 | 0.5000 | 0.0000 | 0.1000 | 0.0000 | <0.01s |

---

## 3. Algorithmic Family Trade-off Analysis

### 3.1 Linear Benchmark (Logistic Regression)
- **Strengths**: Ultra-fast training (0.10s per fold), highly transparent coefficient inspectability, and minimal generalization gap (Train-Val PR-AUC gap of only $+0.0068$).
- **Weaknesses**: Linear hyperplanes cannot capture non-linear feature interactions (such as the combined impact of socioeconomic indicators like `euribor3m` and `poutcome_success` across differing customer age cohorts).
- **Campaign Lens Verdict**: Serves as a solid, trustworthy baseline (PR-AUC 0.4454, ROC-AUC 0.7904). However, it leaves substantial rank-ordering lift on the table compared to tree ensembles.

### 3.2 Bagging Ensemble (Random Forest Classifier)
- **Strengths**: Excels at variance reduction, provides robust minority class identification (highest raw F1-minority score of 0.4796), and parallelizes cleanly across multi-core processors.
- **Weaknesses**: Displays a noticeable train-validation gap ($+0.1131$ to $+0.1328$), indicating an inclination to memorize sparse one-hot encoded sub-spaces even when `max_depth` and `min_samples_leaf` constraints are applied.
- **Campaign Lens Verdict**: High candidate utility, but requires tight regularization to prevent subtle overfitting on low-variance categorical indicators.

### 3.3 Gradient Boosting (HistGradientBoosting Classifier) — *Selected Architecture*
- **Strengths**: Demonstrates the highest cross-validation ranking capability across all models (**Val PR-AUC: 0.4680**, **Val ROC-AUC: 0.8013**). The histogram-based binning algorithm provides native regularization, resulting in an exceptionally well-controlled generalization gap (Train-Val gap of only **$+0.0345$** under tuned parameters `learning_rate=0.04`, `max_leaf_nodes=15`).
- **Weaknesses**: Slightly longer training latency than linear models, though at ~1.58s per fold, it remains extremely practical for recurring batch retraining.
- **Campaign Lens Verdict**: **Best-in-Class Candidate**. Provides monotonic probability calibration and superior lead discrimination within the top score deciles.

### 3.4 Neural Network (Multi-Layer Perceptron MLP)
- **Strengths**: Capable of learning smooth non-linear decision boundaries; produces low Brier score loss (0.0794) under standard loss formulations.
- **Weaknesses**: Sensitive to initial weight states, requires early stopping, and yields poor minority class recall without customized cost-sensitive cross-entropy weighting (F1-minority was only 0.3379).
- **Campaign Lens Verdict**: Inferior to decision tree ensembles for tabular data containing wide one-hot encoded sparse spaces.

---

## 4. Handling of Severe Class Imbalance

1. **Cost-Sensitive Loss Formulations**:
   - Applying `class_weight='balanced'` assigns penalty weights inversely proportional to class frequencies ($w_0 \approx 0.56$, $w_1 \approx 4.44$).
   - This adjusted the loss gradients without introducing synthetic data artifacts or discarding authentic negative records.
2. **Mitigating Resampling Risks**:
   - As established in our research, global SMOTE or under-sampling outside an isolated pipeline leaks distribution statistics across validation folds. Cost-sensitive weighting achieved equivalent sensitivity gains while remaining 100% deterministic and leakage-free.

---

## 5. Probability Output Readiness for Phase 4

For a telemarketing campaign operating under limited outbound call center capacity (e.g., maximum daily call budget of 1,000 to 2,000 leads), **probability rank-ordering is paramount**:
- Both `HistGradientBoosting` and `RandomForest` emit continuous posterior probabilities (`predict_proba`) spanning $[0.0, 1.0]$.
- These probabilities directly support Phase 4 operations:
  - **Threshold Optimization**: Moving beyond the arbitrary 0.5 threshold to optimize business utility (cost per call vs. value of conversion).
  - **Decile Cumulative Gains & Lift Curves**: Validating that dialing the top 20% of model-ranked leads captures over 60–70% of all potential term deposit subscribers.

---

## 6. Verification of Complete Test Set Protection

| Verification Item | Status | Evidence / Audit Log |
|:---|:---:|:---|
| `duration` feature dropped | **PASSED** | Explicitly audited in Cell 2; not present in `X_train_processed.csv`. |
| `X_test_processed.csv` untouched | **PASSED** | Zero calls to test partition files in `03_Model_Development.ipynb`. |
| Cross-Validation isolation | **PASSED** | 5-fold splits partitioned exclusively within `X_train` ($N=32,940$). |
| Deterministic seed pinning | **PASSED** | All estimators and splits executed with `random_state=42`. |
| Artifact serialization | **PASSED** | Models and summaries persisted in `models/candidate_models.joblib`. |
