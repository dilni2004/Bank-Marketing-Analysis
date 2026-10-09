# Preprocessing Decision & Feature Engineering Evidence Log

**Project**: Machine Learning Based Campaign Response Prediction for Bank Marketing  
**Artifact Type**: Canonical Preprocessing & Feature Engineering Evidence Log  
**Canonical File**: `docs/preprocessing_feature_log.md`  
**Associated Notebooks**: [`notebooks/01_Problem_Framing_EDA.ipynb`](../notebooks/01_Problem_Framing_EDA.ipynb), [`notebooks/02_Data_Preprocessing_Pipeline.ipynb`](../notebooks/02_Data_Preprocessing_Pipeline.ipynb)  
**Implementation Source**: [`src/transformers.py`](../src/transformers.py)  
**Authors**: Athukorala M. B. (*Data Engineering*), Siribaddana D. S. (*Problem Framing & EDA*), Nishshanka A. D. N. N. (*Model Development*), Nawarathna M. S. G. (*Evaluation & Diagnostics*)  
**Governance Alignment**: Constitution Principles I–VI, [ADR-001](decision_log.md#adr-001), [ADR-002](decision_log.md#adr-002), [ADR-003](decision_log.md#adr-003), [ADR-004](decision_log.md#adr-004)  
**Status**: Ratified Master Evidence Log  

---

## 1. Executive Summary & Purpose

In accordance with academic assessment criteria (IT3091 Machine Learning) and Constitution Principle VI (*Document As You Build*), this document serves as the **canonical, structured Preprocessing Decision & Feature Engineering Evidence Log**.

Prior to this artifact, several significant data transformations in [`notebooks/02_Data_Preprocessing_Pipeline.ipynb`](../notebooks/02_Data_Preprocessing_Pipeline.ipynb) and [`src/transformers.py`](../src/transformers.py) existed primarily as implementation details in Python code. This log elevates those choices into a formal, transparent, and defensible engineering record.

Every data engineering transformation in the project pipeline:
1. **Evaluates explicit alternatives considered** and articulates specific trade-offs;
2. **Is grounded in quantitative empirical evidence** from [`docs/eda_insight_log.md`](eda_insight_log.md) ($N = 41,188$ records) or operational business logic (pre-call deployment boundaries);
3. **Explicitly distinguishes ratified policies from evolving or tunable choices** (such as the audit and reconciliation of education encoding under ADR-002);
4. **Maintains bidirectional traceability** between exploratory findings, architectural decisions, and production Scikit-Learn transformers;
5. **Is formatted for direct inclusion** in the final academic project report and milestone defense.

---

## 2. Master Structured Preprocessing Decision Table

The table below adheres to the standardized 7-column schema:  
`ID | Decision | Alternatives | Selected | Reason | Evidence | Status`

| ID | Decision | Alternatives | Selected | Reason | Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PRE-01** | Call Duration (`duration`) | Keep as predictor / Bin into proxy intervals / Exclude from operational features | **Remove** | Severe post-event data leakage; call duration is physically unobservable before dialing a customer. Models trained on duration collapse at pre-call scoring. | Business prediction point (pre-call lead qualification); EDA-04 ($r = +0.405$, median $449$s vs $163.5$s); Constitution Principle III. | `Ratified (Leakage Quarantine)` |
| **PRE-02** | Sentinel Value Semantics (`pdays = 999`) | Treat as continuous numeric distance / Impute with mean or median / Treat as special discrete state | **Special state** | 999 is a domain sentinel code signifying uncontacted clients ($96.32\%$), not high metric distance. Imputing or linear scaling imposes spurious Euclidean penalties ($999$ vs $6$ days). | Dataset semantics; EDA-05 ($96.32\%$ uncontacted at $9.26\%$ conversion vs $3.68\%$ valid days at $63.83\%$ conversion); Section 4.6. | `Ratified` |
| **PRE-03** | Numerical Feature Scaling | Uniform `StandardScaler` / Uniform `RobustScaler` / Uniform `MinMaxScaler` / Mixed Scaling | **Mixed Scaling** | Count variables have heavy right tails and extreme outliers (`campaign` max $56$, skew $4.76$; `previous` $86.3\%$ zero-mass), requiring `RobustScaler` (median & IQR). Continuous macroeconomic indicators have wide scale disparities (`nr.employed` $\sim 5000$ vs `euribor3m` $\sim 1-5$) without extreme outliers, requiring `StandardScaler`. | Distribution inspection; skewness & outlier audits; EDA-12, EDA-13, EDA-14, EDA-18; ADR-003, ADR-004. | `Ratified` |
| **PRE-04** | Missing Categorical States (`unknown`) | Complete-case deletion (drop rows) / Mode imputation / Retain as explicit category via One-Hot Encoding | **Retain as explicit category** | Missingness is informative and non-random in CRM records. `default='unknown'` converts at $5.15\%$ vs $12.88\%$ for clean credit; `education='unknown'` converts at $14.50\%$. Dropping rows discards $26\%+$ of the dataset ($10,700+$ records); mode imputation destroys predictive missingness signals. | Unknown frequency analysis; EDA-03 ($N=8,597$ unknown defaults), EDA-08; ADR-002, ADR-003. | `Ratified` |
| **PRE-05** | Educational Attainment Encoding (`education`) | Arbitrary Ordinal Ranking (`unknown`=0, ..., `univ`=7) / Natural Ordinal with Missing Indicator / One-Hot Encoding | **One-Hot Encoding** | Ordinal ranking incorrectly placed `unknown` below `illiterate` and forced monotonic linearity ($w=+0.00513$). Empirical response is non-monotonic (`illiterate`: $22.2\% \to \text{basic.9y}: 7.8\% \to \text{univ}: 13.7\%$). One-Hot Encoding yields $+191$ bps univariate ROC-AUC lift and unconstrained category weights. | $\chi^2 = 193.11, p = 3.31 \times 10^{-38}$; univariate log-loss/ROC-AUC benchmark; multivariate CV benchmark; EDA-08; ADR-002. | `Revised & Ratified (Superseded initial heuristic)` |
| **PRE-06** | Nominal Categorical Encoding | Target/Impact Encoding / Frequency/Count Encoding / One-Hot Encoding (`handle_unknown='ignore'`) | **One-Hot Encoding** | Low-to-moderate cardinalities ($2$ to $12$ levels) yield manageable feature expansion ($+71$ total features, $464:1$ sample-to-feature ratio). Avoids target leakage inherent to target encoding; enables independent coefficient estimation without ordinal metric assumptions. | Cardinality audit; EDA-02, EDA-09, EDA-10, EDA-11, EDA-16; ADR-003, ADR-004. | `Ratified` |
| **PRE-07** | Contact Recency Discretization (`pdays`) | Keep continuous after remapping 999 / Single binary contacted flag / Discrete recency cohorts | **Discrete recency cohorts (`pdays_group`)** | Conversion decays non-linearly with elapsed days: $\le 6$ days converts at $65.8\%$, $7-14$ days at $52.4\%$, $>14$ days at $41.7\%$, uncontacted at $9.26\%$. Binned cohorts (`0_to_6_days`, `7_to_14_days`, `15_plus_days`, `not_contacted`) enable linear models and decision trees to capture the step-function response decay. | Bivariate recency conversion progression; EDA-05; Section 4.6 in EDA log. | `Ratified (Bin cutoffs tunable)` |
| **PRE-08** | Prior Contact Indicator Derivation | Rely exclusively on `previous > 0` / Rely on `poutcome` / Derive binary indicator from `pdays != 999` | **Derive `previously_contacted`** | Provides an explicit, high-signal binary split distinguishing virgin leads ($96.3\%$) from warm re-engagement leads ($3.7\%$). Works in tandem with `previous` count and `poutcome` to isolate the $4,110$ older failed attempts logged with `pdays = 999`. | Cross-field anomaly diagnostics; EDA-05, EDA-06; Section 4.7 in EDA log; ADR-003. | `Ratified` |
| **PRE-09** | Demographic Life-Stage Cohorts (`age`) | Continuous standardized age only / Equal-width interval binning / Domain life-stage cohorts alongside continuous age | **Domain life-stage cohorts (`age_group`)** | Age displays a pronounced U-shaped conversion profile: youth ($<25$: $23.97\%$) and seniors ($60+$: $39.56\%$) convert at elevated rates, while prime workers ($40-49$) convert at only $7.92\%$. Categorical binning (`<30`, `30-39`, `40-49`, `50-59`, `60+`) enables linear estimators to capture non-linear extremes without trimming high-propensity seniors as outliers. | Stratified age KDE; life-stage conversion table; IQR outlier audit; EDA-07; ADR-004. | `Ratified (Cohort cuts tunable)` |
| **PRE-10** | Elimination of Raw Sentinel Column | Retain raw `pdays` alongside engineered features / Impute 999 to 0 and retain / Drop raw `pdays` column | **Drop raw `pdays`** | All domain predictive signals are completely captured by `previously_contacted` and `pdays_group`. Retaining raw `pdays` containing $96.3\%$ sentinel 999 values introduces severe scale distortion in standard scalers and massive spurious Euclidean penalties in distance/gradient models. | Sentinel scale distortion analysis; EDA-05; ADR-003. | `Ratified` |

---

## 3. Detailed Thematic Deep Dives & Empirical Evidence

### 3.1 Post-Call Data Leakage Quarantine (`PRE-01`)

- **Context & Empirical Findings**:  
  Call `duration` exhibits the highest linear correlation with subscription of any individual attribute in the raw dataset ($r = +0.405$). The median call duration for subscribers ($449.0$ seconds) is almost three times higher than that of non-subscribers ($163.5$ seconds). Calls exceeding 10 minutes achieve a conversion rate of $48.61\%$, whereas calls under 1 minute yield an almost zero conversion rate of $0.02\%$ (EDA-04).
- **Core Engineering Implication**:  
  Call duration is determined solely during or after the telemarketing call occurs. At the time of lead qualification and outbound call routing—the operational point of prediction—call duration is unknown. Training models with `duration` creates catastrophic post-event data leakage. The model learns to rely heavily on duration, rendering it entirely useless for pre-call prioritization.
- **Architectural Policy**:  
  In strict adherence to Constitution Principle III (*No Data Leakage*), `duration` is permanently excluded from all predictive pipelines (`remainder='drop'`).

---

### 3.2 Historical Campaign Interaction & Sentinel Discretization (`PRE-02`, `PRE-07`, `PRE-08`, `PRE-10`)

- **Context & Empirical Findings**:  
  - In $96.32\%$ of records ($N = 39,673$), `pdays` contains the sentinel value $999$, converting at $9.26\%$.
  - In $3.68\%$ of records ($N = 1,515$), `pdays` represents valid elapsed days ($[0, 27]$), converting at $63.83\%$ ($6.9\times$ baseline lift).
  - Among valid elapsed days, conversion exhibits steep recency decay: $0-6$ days ($65.8\%$) $\to 7-14$ days ($52.4\%$) $\to >14$ days ($41.7\%$).
  - **Critical Anomaly (EDA-06)**: Exactly $4,110$ records have `previous > 0` but `pdays = 999` (all carry `poutcome = 'failure'`). The legacy CRM defaulted elapsed days to 999 for older failed attempts while logging non-zero `previous` contact counts.
- **Core Engineering Implication**:  
  - Treating 999 as a continuous numeric distance assumes that an uncontacted customer is 166 times "further" than a customer contacted 6 days ago.
  - Relying solely on `pdays != 999` would misclassify $4,110$ warm leads as never contacted.
- **Pipeline Implementation**:  
  `src.transformers.BankFeatureEngineer` executes a three-part transformation:
  1. Derives `previously_contacted = (pdays != 999).astype(int)`;
  2. Discretizes `pdays` into `pdays_group` (`0_to_6_days`, `7_to_14_days`, `15_plus_days`, `not_contacted`);
  3. Drops the raw `pdays` column to eliminate the sentinel 999 from distance and scale calculations, while preserving `previous` count and `poutcome` in downstream transformers.

---

### 3.3 Skewness Management & Mixed Scaling Strategy (`PRE-03`)

- **Context & Empirical Findings**:  
  - `campaign`: Skewness $= 4.76$, kurtosis $= 36.8$, maximum $= 56$ calls, with $99\%$ of records $\le 14$ calls. Conversion drops monotonically from $13.04\%$ on call 1 down to $3.11\%$ on $>10$ calls.
  - `previous`: Skewness $= 3.83$, kurtosis $= 20.1$, maximum $= 7$ calls, with $86.34\%$ zero mass.
  - Macroeconomic indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`): Continuous indicators with severe scale disparity (`nr.employed` ranges $[4963, 5228]$ vs `euribor3m` $[0.634, 5.045]$), but without heavy-tailed outlier distortion.
- **Core Engineering Implication**:  
  Applying `StandardScaler` to `campaign` and `previous` distorts sample variance calculations because extreme outliers pull the variance outward, compressing the bulk of observations into a narrow cluster near zero. Conversely, macroeconomic features require standardization to eliminate scale domination by `nr.employed`.
- **Pipeline Implementation**:  
  A mixed scaling strategy is implemented in `ColumnTransformer`:
  - `RobustScaler` (median centering, IQR scaling) is applied strictly to `campaign` and `previous`.
  - `StandardScaler` (zero mean, unit variance) is applied to all 5 macroeconomic features and continuous `age`.

---

### 3.4 Missingness as an Informative Behavioral Signal (`PRE-04`)

- **Context & Empirical Findings**:  
  Missing values in categorical fields are explicitly coded as `'unknown'`:
  - `default`: $8,597$ records ($20.87\%$) are `'unknown'`, converting at only $5.15\%$ vs $12.88\%$ for clean credit (`no`). Only 3 records have `default = 'yes'` ($0.007\%$, zero conversions).
  - `education`: $1,731$ records ($4.20\%$) are `'unknown'`, converting at $14.50\%$ vs $13.72\%$ for university degree holders.
  - `housing` & `loan`: $990$ records ($2.40\%$) are `'unknown'`.
- **Core Engineering Implication**:  
  In telemarketing operations, `'unknown'` does not reflect missing-at-random (MAR) noise; it represents customer non-disclosure, CRM field omission, or client refusal. Because unobserved credit status yields a distinct, substantially lower conversion rate ($5.15\%$), dropping missing records or imputing them with the mode (`default = 'no'`) destroys a powerful predictive signal and discards over $26\%$ of the training dataset.
- **Pipeline Implementation**:  
  `unknown` is preserved as an explicit, valid category level across all categorical features via `OneHotEncoder(handle_unknown='ignore')`.

---

### 3.5 Education Encoding Evolution: Ordinal Heuristic to Empirical One-Hot Encoding (`PRE-05`, ADR-002)

- **Initial Pipeline Baseline**:  
  The initial data engineering pipeline implemented `OrdinalEncoder` under the heuristic that educational attainment represents a hierarchical progression (`unknown`=0, `illiterate`=1, `basic.4y`=2, ..., `university.degree`=7).
- **Empirical Defects Identified (ADR-002)**:  
  1. *Flawed Missingness Ranking*: Assigning `unknown` to rank 0 declared missingness to be worse than illiteracy, directly contradicting the empirical fact that `unknown` converts at $14.50\%$ (higher than university degree at $13.72\%$).
  2. *Violation of Monotonicity*: The empirical response profile is distinctly U-shaped (`illiterate`: $22.2\%$, `basic.4y`: $10.2\%$, `basic.9y`: $7.8\%$, `high.school`: $10.8\%$, `university.degree`: $13.7\%$). Pearson $\chi^2$ test confirmed strong non-random association ($\chi^2 = 193.11, p = 3.31 \times 10^{-38}$). Ordinal encoding forced a single linear slope ($w = +0.00513$), suppressing predictive power at both extremes.
- **Empirical Lift from One-Hot Encoding**:  
  - Univariate ROC-AUC increased by **$+191$ bps** ($0.5598$ vs $0.5407$).
  - Random Forest 5-fold CV PR-AUC improved from $0.4620$ to **$0.4656$** ($+36$ bps) and held-out test PR-AUC improved from $0.4837$ to **$0.4859$** ($+22$ bps).
  - Dimensionality expanded by only 7 features (from 64 to 71), maintaining an exceptional $464:1$ sample-to-feature ratio on $N_{\text{train}} = 32,950$.
- **Ratified Governance Outcome**:  
  Formalized and ratified in [ADR-002](decision_log.md#adr-002). `education` is permanently moved to `nominal_cols` under `OneHotEncoder(handle_unknown='ignore')`.

---

### 3.6 Nominal Representation & Low-Cardinality Encoding (`PRE-06`)

- **Context & Empirical Findings**:  
  Qualitative features possess low to moderate cardinalities: `contact` (2), `housing` (3), `loan` (3), `default` (3), `marital` (4), `poutcome` (3), `day_of_week` (5), `education` (8), `month` (10), `job` (12).
- **Core Engineering Implication**:  
  Complex encoding schemes such as Target Encoding or CatBoost Encoding introduce significant risk of out-of-fold target leakage, especially with rare levels (`default = 'yes'` has $N = 3$). Frequency encoding discards conditional conversion probabilities. Because total cardinality is low, One-Hot Encoding produces a clean, orthogonal feature space without high-cardinality curse of dimensionality.
- **Pipeline Implementation**:  
  All nominal attributes are encoded via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`. Setting `handle_unknown='ignore'` guarantees inference robustness against unseen categories.

---

### 3.7 Life-Stage Non-Linearity Engineering (`PRE-09`, ADR-004)

- **Context & Empirical Findings**:  
  Overall median age for non-subscribers ($38.0$) and subscribers ($37.0$) is nearly identical, but the subscriber distribution has $40\%$ higher variance ($\sigma = 13.84$ vs $9.90$). Clients under 25 achieve a $23.97\%$ conversion rate, and seniors aged 60+ achieve a $39.56\%$ conversion rate ($>45\%$ for $65+$). Prime working-age adults ($40-49$) convert at only $7.92\%$.
- **Core Engineering Implication**:  
  Standard linear models cannot fit a U-shaped response curve using a single continuous feature ($w \cdot \text{age} \approx 0$). Furthermore, naive outlier trimming based on standard IQR bounds would discard $1,020$ elderly records—the bank's highest-converting demographic segment.
- **Pipeline Implementation**:  
  `BankFeatureEngineer` categorizes `age` into demographic cohorts (`age_group`: `<30`, `30-39`, `40-49`, `50-59`, `60+`) while preserving continuous standardized `age`. This enables linear estimators to assign large positive weights to youth and senior brackets while retaining granular continuous age variation.

---

## 4. Pipeline Code Traceability Matrix

Every documented decision maps 1:1 with code in [`src/transformers.py`](../src/transformers.py) and [`notebooks/02_Data_Preprocessing_Pipeline.ipynb`](../notebooks/02_Data_Preprocessing_Pipeline.ipynb):

| Decision ID | Implementation Class / Component | Module / File Location | Pipeline Step | Input Feature(s) | Output Feature(s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PRE-01** | `ColumnTransformer(remainder='drop')` | `notebooks/02_Data_Preprocessing_Pipeline.ipynb` | `ColumnTransformer` | `duration` | *Excluded (0 features)* |
| **PRE-02** | `BankFeatureEngineer` | `src/transformers.py` | `'feat_eng'` | `pdays` | `previously_contacted`, `pdays_group` |
| **PRE-03** | `RobustScaler` / `StandardScaler` | `notebooks/02_Data_Preprocessing_Pipeline.ipynb` | `'robust'` / `'standard'` | `campaign`, `previous`, macro, `age` | Scaled numeric features |
| **PRE-04** | `OneHotEncoder(handle_unknown='ignore')` | `notebooks/02_Data_Preprocessing_Pipeline.ipynb` | `'nominal'` | Attributes with `'unknown'` | Binary indicator columns (e.g. `default_unknown`) |
| **PRE-05** | `OneHotEncoder(handle_unknown='ignore')` | `notebooks/02_Data_Preprocessing_Pipeline.ipynb` | `'nominal'` | `education` | 8 binary columns (`education_univ...`, `education_unknown`) |
| **PRE-06** | `OneHotEncoder(handle_unknown='ignore')` | `notebooks/02_Data_Preprocessing_Pipeline.ipynb` | `'nominal'` | 10 nominal attributes | Orthogonal indicator columns |
| **PRE-07** | `BankFeatureEngineer` | `src/transformers.py` | `'feat_eng'` | `pdays` | `pdays_group` (`0_to_6_days`, ..., `not_contacted`) |
| **PRE-08** | `BankFeatureEngineer` | `src/transformers.py` | `'feat_eng'` | `pdays` | `previously_contacted` (0 or 1) |
| **PRE-09** | `BankFeatureEngineer` | `src/transformers.py` | `'feat_eng'` | `age` | `age_group` (`<30`, `30-39`, ..., `60+`) |
| **PRE-10** | `BankFeatureEngineer` | `src/transformers.py` | `'feat_eng'` | `pdays` | *Raw pdays column dropped* |

---

## 5. Report Reusability Summary

This master table is formatted for direct transfer into Section 2.2 (*Data Preprocessing & Feature Engineering Design*) of the final academic project report:

```markdown
| ID | Decision | Alternatives Evaluated | Selected Technique | Empirical / Business Rationale | Evidence Base | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| PRE-01 | Duration Quarantine | Keep / Bin / Drop | Remove | Post-call data leakage; unobservable pre-call | Business prediction point / EDA-04 | Ratified |
| PRE-02 | pdays=999 Semantics | Numeric / Impute / Special state | Special state | 999 signifies uncontacted (96.3%), not distance | Dataset semantics / EDA-05 | Ratified |
| PRE-03 | Feature Scaling | Uniform Standard / Robust / Mixed | Mixed Scaling | Heavy skew in counts (max 56); continuous macro | Skewness & IQR audit / EDA-12, 13, 14 | Ratified |
| PRE-04 | Unknown Categories | Drop / Mode / Retain | Retain as category | Informative missingness (default: 5.15% vs 12.88%) | Unknown analysis / EDA-03, 08 | Ratified |
| PRE-05 | Education Encoding | Ordinal / Natural Ord / One-Hot | One-Hot Encoding | Non-monotonic response (illiterate 22.2%, 9y 7.8%) | Chi-sq p=3.3e-38, ADR-002, +191 bps lift | Revised & Ratified |
| PRE-06 | Nominal Encoding | Target / Frequency / One-Hot | One-Hot Encoding | Low cardinality (2-12); avoids target leakage | Cardinality audit / EDA-09, 10, 11 | Ratified |
| PRE-07 | Recency Discretization | Continuous / Binary / Cohorts | Cohorts (pdays_group)| Recency decay (0-6d: 65.8% vs 15+d: 41.7%) | Recency response table / EDA-05 | Ratified |
| PRE-08 | Prior Contact Flag | None / Derive from pdays | previously_contacted | High-signal binary split (63.8% vs 9.3% conversion) | Cross-field audit / EDA-05, 06 | Ratified |
| PRE-09 | Life-Stage Cohorts | Continuous only / Cohorts | age_group + continuous | U-shaped conversion (<25: 24%, 60+: 40%, 40s: 8%) | Stratified age KDE / EDA-07, ADR-004 | Ratified |
| PRE-10 | Drop Raw pdays | Retain / Impute / Drop | Drop raw pdays | Signals captured in groups; prevents scale distortion | Scale distortion analysis / EDA-05 | Ratified |
```
