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

---

## ADR-002: Categorical Encoding Strategy for Education: Selection of One-Hot Encoding over Arbitrary Ordinal Ranking

- **Date**: 2026-10-08
- **Status**: Ratified
- **Authors**: Athukorala M. B. (Data Engineering), Siribaddana D. S. (Problem Framing & EDA), Nishshanka A. D. N. N. (Model Development), Nawarathna M. S. G. (Evaluation & Diagnostics)

### Context & Problem Statement

In the initial implementation of the Phase 2 preprocessing pipeline ([02_Data_Preprocessing_Pipeline.ipynb](file:///Users/moni/Desktop/ML%20proj/notebooks/02_Data_Preprocessing_Pipeline.ipynb)), the demographic variable `education` was treated as an ordinal feature:
```python
ordinal_cols = ['education']
education_hierarchy = [[
    'unknown', 'illiterate', 'basic.4y', 'basic.6y', 'basic.9y',
    'high.school', 'professional.course', 'university.degree'
]]
preprocessor = ColumnTransformer(
    transformers=[
        ...
        ('ordinal', OrdinalEncoder(categories=education_hierarchy, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_cols),
        ...
    ]
)
```

A critical architectural review identified two severe methodological and empirical defects in this configuration:
1. **Misclassification of Missingness (`unknown`) as an Ordinal Education Level**:
   The category `unknown` was placed at index position 0, assigning it an ordinal rank strictly lower than `illiterate` (position 1). In domain and measurement theory, `unknown` denotes unobserved or unspecified customer information (item non-response, client privacy preference, or logging omission)—not an educational attainment milestone. Assigning `unknown` to the lowest rank ($0.0$) declares missingness to be worse than illiteracy.
   Moreover, empirical analysis reveals that clients with `unknown` education exhibit a positive response rate of **$14.50\%$** ($251$ subscribers out of $1,731$ records). This conversion rate is higher than that of `university.degree` holders ($13.72\%$) and nearly double that of `basic.9y` clients ($7.82\%$). Forcing `unknown` to the minimum scalar value ($0.0$) compels linear models to assume the lowest linear response propensity for unobserved records, directly contradicting empirical data.
2. **Imposition of Artificial Linearity, Monotonicity, and Spurious Metric Distance**:
   Applying `OrdinalEncoder` assigns integer values $\{0, 1, 2, \dots, 7\}$, which imposes an interval-scale assumption with uniform Euclidean distance:
   $$\text{dist}(\text{illiterate}, \text{basic.4y}) = \text{dist}(\text{basic.4y}, \text{basic.6y}) = \text{dist}(\text{high.school}, \text{professional.course}) = 1.0$$
   This implies that advancing from primary school to lower secondary school carries the identical metric interval and behavioral impact as transitioning from high school to professional/vocational training.
   More critically, the empirical conversion rate across educational levels is **distinctly non-monotonic**:
   `illiterate` ($22.22\%$) $\rightarrow$ `basic.4y` ($10.25\%$) $\rightarrow$ `basic.6y` ($8.20\%$) $\rightarrow$ `basic.9y` ($7.82\%$) $\rightarrow$ `high.school` ($10.84\%$) $\rightarrow$ `professional.course` ($11.35\%$) $\rightarrow$ `university.degree` ($13.72\%$) $\rightarrow$ `unknown` ($14.50\%$).
   For linear models (Logistic Regression, Linear SVM, Perceptron, Neural Networks), an ordinal feature forces a single monotonic weight slope $w \cdot x$. This mathematically prevents linear classifiers from capturing the high conversion rates at both the lowest attainment tier (`illiterate`: $22.2\%$) and highest attainment tier (`university.degree`: $13.7\%$) without being penalized by intermediate troughs (`basic.9y`: $7.8\%$).
3. **Pipeline Inconsistency with Other Demographic Categorical Features**:
   All other demographic categorical attributes containing `unknown` values (`job`, `marital`, `default`, `housing`, `loan`) were modeled as nominal variables via `OneHotEncoder(handle_unknown='ignore')`, creating discrete binary indicator columns (e.g. `job_unknown`, `marital_unknown`). Isolating `education` for ordinal ranking broke architectural uniformity without theoretical or empirical justification.

---

### Options Considered

#### Option A – Current Ordinal Encoding (`unknown` at Position 0)
- **Description**: Map categories to integer scalars: `unknown`=0, `illiterate`=1, `basic.4y`=2, `basic.6y`=3, `basic.9y`=4, `high.school`=5, `professional.course`=6, `university.degree`=7.
- **Evaluation & Fatal Flaws**:
  - Confounds missing data with lowest educational attainment.
  - Imposes equal spacing on non-metric qualitative categories.
  - Constrains linear models to a single monotonic slope ($w = +0.00513$), causing severe probability miscalibration on `illiterate` and `unknown` segments.
  - Limits tree algorithms to contiguous range splits ($x \le c$), requiring deep, multi-level splits to isolate non-adjacent high-converting segments.

#### Option B – One-Hot Encoding (Nominal Representation, `unknown` as Distinct Category)
- **Description**: Include `education` in `nominal_cols` along with other demographic features, transforming it via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` into 8 orthogonal binary indicator columns:
  `education_basic.4y`, `education_basic.6y`, `education_basic.9y`, `education_high.school`, `education_illiterate`, `education_professional.course`, `education_university.degree`, `education_unknown`.
- **Empirical Advantages**:
  - **Zero Arbitrary Ranking**: `unknown` is treated strictly as an unobserved categorical state (`education_unknown = 1`), eliminating false educational ordering.
  - **Full Non-Monotonic Capacity**: Each education tier receives an independent, unconstrained weight/coefficient in linear models ($w_{\text{illiterate}} = +0.222$, $w_{\text{basic.9y}} = -0.187$, $w_{\text{unknown}} = -0.030$), accommodating the empirical U-shaped response curve.
  - **Orthogonal Tree Partitioning**: Tree-based ensembles (Random Forest, Gradient Boosting) can isolate any individual category in a single split without range-adjacency constraints.
  - **Architectural Uniformity**: Aligns with the encoding strategy applied across all other categorical features with missingness (`job`, `marital`, `default`, `housing`, `loan`).
  - **Negligible Dimensionality Impact**: Adds only 7 net features (expanding total output from 64 to 71 features). Given $N_{\text{train}} = 32,950$, the sample-to-feature ratio is $464:1$, far above any threshold for overfitting.

#### Option C – Natural Ordinal Hierarchy + Separate Missingness Indicator
- **Description**: Assign genuine educational levels to integers (`illiterate`=0 to `university.degree`=6), impute `unknown` to the median (or mode), and append a binary indicator column `education_is_unknown`.
- **Evaluation**: While resolving the `unknown` ranking flaw, it preserves the flawed equal-interval and monotonicity constraints across the 7 genuine tiers (forcing `illiterate` $\le$ `basic.4y` $\le \dots \le$ `university.degree`), which fails to match the empirical non-monotonicity. Option C adds pipeline complexity without matching the representational freedom or uniformity of Option B.

---

### Empirical Evidence & Quantitative Benchmark

#### 1. Category Distribution & Empirical Conversion Profile

| Education Category | Count ($N$) | Proportion (%) | Subscribed ($y=1$) | Empirical Rate (%) | OHE Logit Weight ($w_k$) | Option A Assigned Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `unknown` | 1,731 | 4.20% | 251 | **14.50%** | -0.02987 | Rank 0 (Flawed) |
| `illiterate` | 18 | 0.04% | 4 | **22.22%** | **+0.22202** | Rank 1 |
| `basic.4y` | 4,176 | 10.14% | 428 | 10.25% | -0.14158 | Rank 2 |
| `basic.6y` | 2,292 | 5.56% | 188 | 8.20% | +0.03454 | Rank 3 |
| `basic.9y` | 6,045 | 14.68% | 473 | 7.82% | -0.18660 | Rank 4 |
| `high.school` | 9,515 | 23.10% | 1,031 | 10.84% | -0.11042 | Rank 5 |
| `professional.course` | 5,243 | 12.73% | 595 | 11.35% | -0.11854 | Rank 6 |
| `university.degree` | 12,168 | 29.54% | 1,670 | 13.72% | -0.02623 | Rank 7 |

*Statistical Significance*: Pearson $\chi^2$ test of independence confirms strong non-random association with subscription: $\chi^2 = 193.11$, $p = 3.31 \times 10^{-38}$ ($\text{df}=7$).

#### 2. Univariate Predictive Power
- **Option A (Ordinal Encoder, unknown=0)**: ROC-AUC = **$0.5407$**, Log-Loss = **$0.35151$**
- **Option C (Natural Ordinal + Missing Indicator)**: ROC-AUC = **$0.5543$**, Log-Loss = **$0.35043$**
- **Option B (One-Hot Encoding)**: ROC-AUC = **$0.5598$**, Log-Loss = **$0.34965$**

*Finding*: One-Hot Encoding achieves a **+191 bps lift** in univariate ROC-AUC and lower log-loss over the ordinal representation, confirming that unconstrained categorical representation better preserves predictive signal.

#### 3. Multivariate 5-Fold Stratified Cross-Validation & Held-out Test Benchmarks

| Model Family | Validation Metric | Option A: Current Ordinal (64 feats) | Option B: One-Hot Encoding (71 feats) | Delta / Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 5-Fold CV ROC-AUC | $0.7891 \pm 0.0059$ | $0.7890 \pm 0.0064$ | Parity ($-0.0001$) |
| | 5-Fold CV PR-AUC | $0.4490$ | $0.4487$ | Parity ($-0.0003$) |
| | Held-out Test ROC-AUC | $0.8019$ | $0.8013$ | Parity ($-0.0006$) |
| | Held-out Test PR-AUC | $0.4672$ | $0.4666$ | Parity ($-0.0006$) |
| **Random Forest** | 5-Fold CV ROC-AUC | $0.7954 \pm 0.0039$ | **$0.7961 \pm 0.0031$** | **+0.0007** (Lower variance) |
| | 5-Fold CV PR-AUC | $0.4620$ | **$0.4656$** | **+0.0036** (+36 bps lift) |
| | Held-out Test ROC-AUC | $0.8110$ | $0.8102$ | Parity ($-0.0008$) |
| | Held-out Test PR-AUC | $0.4837$ | **$0.4859$** | **+0.0022** (+22 bps lift) |
| **HistGradientBoosting** | 5-Fold CV ROC-AUC | $0.7982 \pm 0.0032$ | $0.7976 \pm 0.0041$ | Parity ($-0.0006$) |
| | 5-Fold CV PR-AUC | $0.4667$ | $0.4664$ | Parity ($-0.0003$) |
| | Held-out Test ROC-AUC | $0.8113$ | $0.8095$ | Parity ($-0.0018$) |
| | Held-out Test PR-AUC | $0.4882$ | **$0.4891$** | **+0.0009** (+9 bps lift) |

---

### Decision Outcome

**Ratified Decision**: Adopt **Option B (One-Hot Encoding of Education)** as the primary categorical encoding strategy across the project. Discontinue and abandon the `OrdinalEncoder` hierarchy for `education`.

**Implementation Actions**:
1. Move `'education'` from `ordinal_cols` to `nominal_cols` in `ColumnTransformer`.
2. Remove `education_hierarchy` and the `('ordinal', ...)` step from `ColumnTransformer`.
3. Expand the nominal feature pipeline via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` to produce 8 distinct indicator features:
   `education_basic.4y`, `education_basic.6y`, `education_basic.9y`, `education_high.school`, `education_illiterate`, `education_professional.course`, `education_university.degree`, `education_unknown`.
4. Treat `unknown` strictly as an orthogonal categorical state (`education_unknown = 1`), eliminating its flawed placement below `illiterate`.
5. Update pipeline output dimensionality from 64 to 71 features.
6. Re-generate all serialized artifacts (`X_train_processed.csv`, `X_test_processed.csv`, `preprocessor_pipeline.joblib`, `feature_metadata.json`).

---

### Scope Reconciliation & Governance Alignment

1. **Reconciliation Statement**:
   This decision formally supersedes the initial ordinal encoding assumption in Phase 2 Data Engineering. Preprocessing artifacts and downstream modeling phases will utilize the 71-feature schema established by Option B.
2. **Constitution Compliance**:
   - **Principle I (Reproducibility First)**: All transformations execute deterministically with explicit seeds (`random_state=42`).
   - **Principle II (Pipeline-First)**: Transformation logic resides strictly inside Scikit-Learn's `ColumnTransformer` / `Pipeline` architecture.
   - **Principle III (No Data Leakage)**: Category vocabularies are learned strictly on `X_train` and applied to `X_test`.
   - **Principle IV (Evaluation Honesty)**: Comprehensive cross-validation and test metrics across multiple model families accompany this decision.
---

## ADR-003: Comprehensive Data Quality Invariants, Cross-Field Consistency, and Data Leakage Controls

- **Date**: 2026-10-08
- **Status**: Ratified
- **Authors**: Athukorala M. B. (Data Engineering), Siribaddana D. S. (Problem Framing & EDA), Nishshanka A. D. N. N. (Model Development), Nawarathna M. S. G. (Evaluation & Diagnostics)

### Context & Problem Statement

Initial exploratory data analysis covered baseline descriptive checks (dataset shape, column types, null counts, `unknown` category frequencies, and target class imbalance). However, an academically rigorous machine learning project requires deep exploration into data quality anomalies, measurement conventions, and operational characteristics that directly dictate preprocessing and modelling decisions (in alignment with the IT3091 Machine Learning rubric).

Specifically, eight critical data quality dimensions required rigorous auditing and formal policy determination:
1. Exact and feature-subset duplicate records;
2. Numerical domain range validation;
3. Categorical cardinality and calendar constraints;
4. Unusually rare categories and small-sample instability;
5. Distribution skewness and outlier handling;
6. Sentinel value semantics (`pdays = 999`);
7. Cross-field consistency between historical campaign fields (`previous`, `pdays`, `poutcome`);
8. Post-event data leakage (`duration`).

---

### Empirical Findings Summary

| Dimension / Check | Empirical Evidence | Risk / Modelling Impact | Policy & Preprocessing Decision |
| :--- | :--- | :--- | :--- |
| **Exact Duplicates** | Exactly 12 duplicate rows ($0.029\\%$), zero conflicting labels across duplicate feature profiles | Train/test split leakage; objective function overweighting | Prune exact duplicate rows prior to splitting (`df.drop_duplicates()`), preserving 41,176 distinct records |
| **Numerical Range Validation** | All 10 numerical features satisfy domain bounds; zero illegal negative entries; `cons.conf.idx` entirely negative ($-50.8$ to $-26.9$) | Disparate scales (`nr.employed` $\\sim 5000$ vs `euribor3m` $\\sim 1-5$) distort distance and gradient methods | Apply `StandardScaler` to macroeconomic variables and age; apply `RobustScaler` to heavy-tailed counts |
| **Categorical Cardinality** | Low-to-moderate cardinalities ($2$ to $12$); 10 months (Jan/Feb absent); 5 weekdays (Sat/Sun absent) | Combinatorial expansion if high; unobserved categories crashing inference | One-Hot Encoding for nominal variables; configure `handle_unknown='ignore'` for production resilience |
| **Rare Categories** | `default='yes'` ($N=3$, $0.007\\%$), `illiterate` ($N=18$, $0.044\\%$), `dec` ($N=182$, $0.442\\%$) | Zero-sample splits in k-fold CV; erratic weights in unpenalized models | Enforce L2 regularization; interpret `default` primarily as a missingness indicator rather than credit risk |
| **Outliers & Heavy Tails** | Extreme right skew in `campaign` (max $56$) and `duration` (max $4,918$s); moderate skew in `age` (max $98$) | Distorted variance estimates; risk of discarding high-propensity seniors | Retain elderly records (conversion $> 45\\%$); apply `RobustScaler` to `campaign` and `previous` |
| **Sentinel Value Semantics** | `pdays = 999` in $96.32\\%$ of records ($N=39,673$); valid elapsed days in $3.68\\%$ ($[0, 27]$ days) | Treating 999 continuously forces arbitrary Euclidean distance and invalid linear slopes | Feature engineer `previously_contacted` binary indicator and `pdays_group` categorical bins; drop raw `pdays` |
| **Cross-Field Consistency** | $4,110$ records have `previous > 0` but `pdays = 999` (all $4,110$ carry `poutcome = 'failure'`) | Relying solely on `pdays != 999` misclassifies $73.07\\%$ of previously contacted clients as never contacted | Define prior interaction by `previous > 0`; retain both `previous` and `poutcome` in pipeline |
| **Post-Event Data Leakage** | `duration` strongly correlates with $y$ ($r = +0.405$); median $449$s for subscribers vs $163.5$s for non-subscribers | Duration is unobservable before placing calls; models collapse in pre-call deployment | Strictly drop `duration` from operational predictive feature pipeline (`remainder='drop'`) |

---

### Ratified Architectural Policies

1. **Deduplication Policy**: Exact duplicates are removed prior to train-test splitting in Phase 2 to prevent identical customer profiles from leaking across partitions.
2. **Sentinel Transformation Policy**: The raw `pdays` column is never supplied directly to scale-dependent or linear estimators. It is transformed into a binary contact flag and discrete recency cohorts via `src.transformers.BankFeatureEngineer`.
3. **Historical Campaign Feature Preservation**: To prevent loss of information regarding older campaign attempts ($4,110$ failed contacts logged with `pdays = 999`), both `previous` (count of prior contacts) and `poutcome` (historical campaign result) are preserved in the pipeline.
4. **Data Leakage Quarantine Policy**: In strict compliance with Constitution Principle III (*No Data Leakage*), `duration` is excluded from all candidate model training and production evaluation pipelines.


