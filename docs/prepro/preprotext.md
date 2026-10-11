# Data Preprocessing Pipeline - Text & Comments

> Extracted text blocks (markdown cells) and code comments from [`notebooks/02_Data_Preprocessing_Pipeline.ipynb`](file:///Users/moni/Desktop/ML proj/notebooks/02_Data_Preprocessing_Pipeline.ipynb).

---

# Phase 2: Data Preprocessing & Leakage-Free Feature Pipeline

**Project**: Machine Learning Based Campaign Response Prediction for Bank Marketing  
**Phase**: 02 – Data Engineering & Preprocessing  
**Owner**: Athukorala M. B. (IT24101937)  
**Governance & Standards**: Aligned with Constitution Principles I–VI, [ADR-001 (Validation Split Strategy)](../docs/decision_log.md#adr-001), [ADR-002 (Categorical Encoding Strategy for Education)](../docs/decision_log.md#adr-002), and [ADR-004 (Dual Age Representation)](../docs/decision_log.md#adr-004)

### Purpose
This notebook implements the reproducible, leakage-free data engineering and feature preprocessing pipeline for the UCI Bank Marketing dataset (`bank-additional-full.csv`). It resolves the validation split strategy discrepancy via an empirical investigation (ADR-001), rectifies the encoding of customer educational attainment by replacing arbitrary ordinal ranking with one-hot encoding (ADR-002), establishes strict fit-on-train data isolation, applies modular Scikit-Learn transformers, and exports serialized artifacts for downstream model training.

### Inputs
- Raw Dataset: `data/raw/bank-additional-full.csv` (41,188 records, 20 features + target `y`)
- Modular Transformer: `src.transformers.BankFeatureEngineer`

### Outputs
- `data/processed/X_train_processed.csv`: Transformed training feature matrix (32,950 × 71)
- `data/processed/X_test_processed.csv`: Transformed holdout test feature matrix (8,238 × 71)
- `data/processed/y_train.csv`: Training target vector (32,950 records)
- `data/processed/y_test.csv`: Holdout target vector (8,238 records)
- `data/processed/preprocessor_pipeline.joblib`: Serialized fitted `Pipeline`
- `data/processed/feature_metadata.json`: Feature catalog and pipeline specifications (71 features)

> **Code Comments (Cell 1):**
> - Deterministically resolve project root independent of caller working directory
> - Plotting & display configuration

## 1. Raw Data Ingestion, Deduplication (ADR-003), & Target Isolation (PRE-01)

In strict adherence to **Constitution Principle I (*Reproducibility First*)**, the raw dataset located in `data/raw/bank-additional-full.csv` is treated as strictly read-only.

In accordance with **[ADR-003](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-003)** and **[EDA-17](file:///Users/moni/Desktop/ML%20proj/docs/eda_insight_log.md)**, exact duplicate rows are audited and pruned prior to splitting (`df_raw.drop_duplicates()`), preserving exactly **41,176 distinct customer records** and eliminating cross-split duplicate leakage.

In accordance with **PRE-01** and **Constitution Principle III (*No Data Leakage*)**, the target variable `y` is encoded as binary (0/1), while call `duration` is quarantined and permanently removed from the feature matrix because it is physically unobservable before placing a call.

> **Code Comments (Cell 3):**
> - Locate dataset dynamically using PROJECT_ROOT anchor
> - ADR-003 & EDA-17 Deduplication Policy: Prune exact duplicate rows prior to splitting

> **Code Comments (Cell 4):**
> - 1. Target encoding: 'yes' -> 1, 'no' -> 0
> - 2. PRE-01 / ADR-003: Drop target and post-event leakage variable 'duration'

## 2. Validation Split Strategy Investigation & Rationale

### Scope Discrepancy Context
The initial project submission proposed a validation methodology that considers the chronological ordering of the campaign data to avoid leaking time-related patterns into holdout sets. However, before establishing our production preprocessing pipeline, we conducted an empirical investigation comparing:
- **Option A – Temporal Holdout**: Partitioning the dataset chronologically (first 80% development, last 20% holdout).
- **Option B – Stratified Random Holdout**: Partitioning via stratified random sampling (80% development, 20% holdout with `random_state=42`).

The comparison below details why Option B is retained and formally ratified in `docs/decision_log.md` (ADR-001).

> **Code Comments (Cell 6):**
> - Empirical Investigation: Option A (Temporal Split) vs Option B (Stratified Random Split)
> - Option A: Chronological 80/20 split
> - Option B: Stratified 80/20 split (for comparison)
> - 1. Target class distribution comparison
> - 2. Covariate Shift Comparison (Macroeconomic Indicators)
> - 3. Categorical Support Coverage

### Summary of Findings & Decision (ADR-001 & ADR-003)
As demonstrated by the empirical comparison above and ratified under **[ADR-001](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-001)**:
- **Option A (Temporal Split)** introduces severe macroeconomic regime shifts ($euribor3m$ zero support overlap) and an extreme $+384\%$ shift in positive base rate ($6.37\%$ to $30.83\%$), rendering model cross-validation and probability calibration unviable.
- **Option B (Stratified Random Split)** is ratified as the primary partitioning strategy with explicit seed `random_state=42`. Combining Option B with **[ADR-003 Deduplication](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-003)** ensures identical $11.27\%$ priors, uniform covariate coverage, and zero duplicate leakage across partitions.

> **Code Comments (Cell 8):**
> - Ratified Execution: Stratified 80/20 train/test split with contract verification (ADR-001 & ADR-003)
> - Contract Invariant Assertions (ADR-001 & ADR-003 Reconciled: 41,176 records)
> - Mutual Exclusivity Assertion
> - Cross-Split Duplicate Profile Leakage Assertion (Zero duplicate profiles across partitions)
> - Stratification Balance Assertion (< 0.1% delta)

## 3. Modular Feature Engineering (`src.transformers`)

In adherence to **Constitution Principle II (*Pipeline-First*)**, domain feature transformations are encapsulated in [`src/transformers.py`](../src/transformers.py) via `BankFeatureEngineer`:
- **PRE-02**: Treating sentinel `pdays = 999` ($96.32\%$ uncontacted) as a discrete behavioral state rather than a continuous metric distance.
- **PRE-07**: Discretizing elapsed days into non-linear recency cohorts (`pdays_group`: `0_to_6_days`, `7_to_14_days`, `15_plus_days`, `not_contacted`).
- **PRE-08**: Deriving binary interaction indicator `previously_contacted` ($pdays \ne 999$).
- **PRE-09 & ADR-004**: Life-stage non-linearity engineering via categorical `age_group` (`<30`, `30-39`, `40-49`, `50-59`, `60+`) while intentionally preserving continuous standardized `age` (dual representation).
- **PRE-10**: Dropping raw `pdays` column to eliminate sentinel 999 from distance and scale computations.
- **ADR-003**: Preserving historical campaign features (`previous` count and `poutcome` status) to retain information regarding the 4,110 legacy CRM failure contacts.

> **Code Comments (Cell 10):**
> - Ensure project root is in sys.path for modular imports
> - Deterministically ensure PROJECT_ROOT is defined before use
> - Test feature engineer transformer on X_train only

## 4. Categorical Encoding Strategy: Education Feature Representation (ADR-002, PRE-05)

Initial data engineering heuristic treated `education` as an ordinal feature (`unknown`=0, `illiterate`=1, ..., `university.degree`=7). As evaluated in **[ADR-002](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-002)** and **[PRE-05](file:///Users/moni/Desktop/ML%20proj/docs/preprocessing_feature_log.md)**, this was defective because:
1. It conflated missingness (`unknown`) with lowest attainment;
2. It violated the non-monotonic empirical conversion profile (`illiterate`: $22.2\%$, `basic.9y`: $7.8\%$, `univ`: $13.7\%$, `unknown`: $14.5\%$).

> **Code Comments (Cell 12):**
> - Empirical Investigation: Categorical Encoding Strategies for Education
> - 1. Demographic Distribution & Empirical Response Rates
> - 2. Univariate Predictive Capacity (Ordinal vs OHE)
> - 3. Multivariate Model Comparison Across Pipeline Candidates
> - Pipeline A: Current Ordinal
> - Pipeline B: One-Hot Encoding
> - Compare Logistic Regression Coefficients on Education
> - 4. Cross-Validation and Holdout Benchmark Across Model Families

### Summary of Findings & Decision (ADR-002 & PRE-05)

As ratified under **[ADR-002](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-002)**, the arbitrary ordinal hierarchy for `education` is completely replaced by **One-Hot Encoding (Option B)** across all 8 attainment levels (including `unknown` as an orthogonal state). This yields $+191$ bps univariate ROC-AUC lift and unconstrained coefficient estimation across linear and tree models.

## 4.5 Project-Wide Policy for "Unknown" Categorical States (PRE-04, ADR-003)

**Ratified Handling Policy (PRE-04 & ADR-003)**:
- **No rows are silently dropped** because of `"unknown"`. Dropping records would discard $>26\%$ of the dataset.
- `"unknown"` is explicitly retained as a separate valid category across all nominal variables (`job`, `marital`, `default`, `housing`, `loan`, `education`). Missingness in CRM data is informative (e.g. `default='unknown'` converts at $5.15\%$ vs $12.88\%$ for clean credit).
- All nominal attributes are passed through `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

## 5. End-to-End Scikit-Learn Preprocessing Pipeline Architecture (PRE-01..10, ADR-001..004)

The production preprocessing pipeline integrates all ratified architectural decisions into a single Scikit-Learn `Pipeline`:
- **PRE-01**: `duration` quarantined via `remainder='drop'` (zero post-event target leakage);
- **PRE-02, PRE-07, PRE-08, PRE-09, PRE-10**: Engineered domain features generated via `BankFeatureEngineer`;
- **PRE-03**: Mixed scaling applying `RobustScaler` to heavy-tailed counts (`campaign`, `previous`) and `StandardScaler` to macroeconomic variables and continuous `age`;
- **PRE-04, PRE-05, PRE-06, PRE-09**: Nominal categorical features encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` producing orthogonal binary indicator columns;
- Total pipeline output dimensionality: **71 orthogonal features**.

> **Code Comments (Cell 16):**
> - 1. Feature column definitions
> - PRE-04, PRE-05, PRE-06, PRE-07, PRE-09: Nominal One-Hot Encoded columns
> - PRE-05 / ADR-002: 8 binary indicator columns
> - PRE-09 / ADR-004: 5 life-stage cohorts (<30, 30-39, 40-49, 50-59, 60+)
> - PRE-07: 4 recency cohorts (0_to_6_days, 7_to_14_days, 15_plus_days, not_contacted)
> - PRE-03: Heavy-tail robust scaling for count variables with extreme outliers
> - PRE-03 & PRE-09 (ADR-004): Standardization for macroeconomic indicators and continuous age
> - PRE-09: Continuous standardized age preserved for fine-grained ranking
> - PRE-08: Binary prior contact flag passthrough
> - 2. Build ColumnTransformer (PRE-01, PRE-03, PRE-04, PRE-05, PRE-06, PRE-09)
> - PRE-01: Quarantines post-event duration leakage
> - 3. Assemble full preprocessing pipeline (PRE-01..PRE-10, ADR-001..ADR-004)

## 6. Strict Fit-on-Train Pipeline Execution (Zero Data Leakage)
In compliance with Constitution Principle III (*No Data Leakage*), all transformer scaling parameters ($\mu, \sigma$), categorical one-hot dictionaries, and grouping statistics are fitted strictly on `X_train`. The holdout set `X_test` is transformed using the fitted pipeline without parameter re-estimation.

> **Code Comments (Cell 18):**
> - Fit on training data ONLY
> - Transform train and holdout test sets
> - Extract column names from fitted ColumnTransformer
> - Create clean DataFrames
> - Contract assertions (ADR-002: 71 features; ADR-003: 32,940 train / 8,236 test)

## 7. Post-Scaling Feature Distributions Inspection

> **Code Comments (Cell 20):**
> - Diagnostic Plots
> - 1. Age distribution standardized
> - 2. Campaign distribution robust-scaled
> - 3. Euribor 3-month rate standardized

## 8. Exporting Processed Dataset Artifacts & Pipeline Serialization
All processed datasets are saved to `data/processed/`, accompanied by the fitted pipeline object and metadata schema reflecting ADR-001 and ADR-002.

> **Code Comments (Cell 22):**
> - Create output directory anchored to PROJECT_ROOT
> - 1. Save processed feature sets
> - 2. Serialize full fitted pipeline
> - 3. Export feature catalog & metadata schema (ADR-001..ADR-004, PRE-01..PRE-10)

## 9. Serialization & Inference Verification

> **Code Comments (Cell 24):**
> - Reload serialized pipeline
> - Verify estimator parameters inspection (contract requirement)
> - Take 5 raw samples from X_test (before any transformation)
> - Transform using loaded pipeline
> - Verify exact numerical match with original transformed test set

