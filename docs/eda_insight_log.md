# Exploratory Data Analysis (EDA) Insight & Evidence Log

**Project**: Machine Learning Based Campaign Response Prediction for Bank Marketing  
**Artifact Type**: Core Empirical Evidence Log  
**Canonical File**: `docs/eda_insight_log.md`  
**Associated Notebook**: [`notebooks/01_Problem_Framing_EDA.ipynb`](file:///Users/moni/Desktop/ML%20proj/notebooks/01_Problem_Framing_EDA.ipynb)  
**Authors**: Siribaddana D. S. (*Problem Framing & EDA*), Athukorala M. B. (*Data Engineering*), Nishshanka A. D. N. N. (*Model Development*), Nawarathna M. S. G. (*Evaluation & Diagnostics*)  
**Status**: Ratified Core Evidence  

---

## 1. Executive Summary & Purpose

In accordance with project assessment criteria and Constitution Principle VI (*Document As You Build*), this document serves as the **canonical, structured EDA Insight Log**. It bridges initial exploratory observations from [`notebooks/01_Problem_Framing_EDA.ipynb`](file:///Users/moni/Desktop/ML%20proj/notebooks/01_Problem_Framing_EDA.ipynb) directly to downstream machine learning engineering decisions in [`src/transformers.py`](file:///Users/moni/Desktop/ML%20proj/src/transformers.py), [`notebooks/02_Data_Preprocessing_Pipeline.ipynb`](file:///Users/moni/Desktop/ML%20proj/notebooks/02_Data_Preprocessing_Pipeline.ipynb), and architectural policies documented in [`docs/decision_log.md`](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md).

Every observation recorded here is grounded in **quantitative evidence** from the dataset ($N = 41,188$ records) and carries an explicit **machine learning implication** governing feature representation, scaling, leakage quarantine, loss formulation, or evaluation design.

---

## 2. Master Structured EDA Insight Log

The table below follows the standardized five-column evidence schema:
`ID | Variable(s) | Observation | Evidence | ML implication`

| ID | Variable(s) | Observation | Evidence | ML implication |
| :--- | :--- | :--- | :--- | :--- |
| **EDA-01** | `y` | Severe class imbalance; positive response rate is $11.27\%$ ($4,640$ subscribers vs $36,548$ non-subscribers). | Target class distribution: $88.73\%$ `no` vs $11.27\%$ `yes` (Section 2, Cells 13–15). | Avoid accuracy-only evaluation; adopt PR-AUC, ROC-AUC, F1-score, and cost-weighted utility; enforce stratified train/test partitioning. |
| **EDA-02** | `poutcome` | Previous campaign success is strongly associated with response: $65.11\%$ conversion ($5.8\times$ baseline lift). Even prior failure converts at $14.23\%$, outperforming virgin cold leads ($8.83\%$). | Historical response rate analysis: $N=1,373$ success ($65.11\%$), $N=4,252$ failure ($14.23\%$), $N=35,563$ nonexistent ($8.83\%$) (Section 3.3, Cells 27–29). | Prime candidate predictor; apply One-Hot Encoding; require cold-lead subgroup diagnostics (`previous=0`) to ensure models do not degenerate into prior-success lookups. |
| **EDA-03** | `default` | Large `unknown` credit status category ($8,597$ records, $20.87\%$) converts at only $5.15\%$ vs $12.88\%$ for clean credit (`no`). Default `yes` has only 3 cases ($0.007\%$) with 0 conversions. | Unknown frequency analysis: $N=8,597$ ($20.87\%$) unknown; contingency table shows $5.15\%$ vs $12.88\%$ conversion (Section 1.2 Cell 10, Section 3.1 Cell 21, Section 4.4 Cell 47). | Retain `unknown` as an explicit categorical state via OHE; treat feature primarily as missing credit record indicator rather than credit risk; enforce L2 regularization to prevent weight explosion on rare `yes`. |
| **EDA-04** | `duration` | Highly predictive ($r=+0.405$, median $449$s subscribers vs $163.5$s non-subscribers; calls $>10$m convert at $48.61\%$ vs $0.02\%$ for $\le 1$m), but call duration is unobservable before placing a call. | Correlation $r=+0.405$, KDE duration shift, quartile response rates (Section 4.8, Cells 59–62). | Remove entirely from operational feature pipeline (`remainder='drop'`) due to post-event data leakage; models trained on duration collapse at pre-call scoring. |
| **EDA-05** | `pdays` | $96.32\%$ of records ($N=39,673$) carry sentinel value $999$ (uncontacted), converting at $9.26\%$. Only $3.68\%$ ($N=1,515$) have valid elapsed days ($[0, 27]$), converting at $63.83\%$. | Sentinel distribution analysis: $96.32\%$ at 999; valid days mean $6.01$, median $6.0$; $6.9\times$ conversion gap (Section 4.6, Cells 52–55). | Do not treat 999 as continuous metric distance; engineer binary flag `previously_contacted` (`pdays != 999`) and recency bins `pdays_group`; drop raw `pdays` column. |
| **EDA-06** | `previous`, `pdays`, `poutcome` | $4,110$ records have `previous > 0` but `pdays = 999` (all carry `poutcome = 'failure'`). The telemarketing CRM only logged exact day counts for recent/successful contacts, defaulting older failed contacts to 999. | Cross-field consistency diagnostics: $100\%$ of the $4,110$ anomalous records have `poutcome='failure'` (Section 4.7, Cells 56–58). | Flagging prior contact solely by `pdays != 999` misclassifies $4,110$ clients ($73\%$ of contacted). Must retain both `previous` count and `poutcome` in the feature pipeline. |
| **EDA-07** | `age` | U-shaped conversion: clients $<25$ convert at $23.97\%$ and seniors $60+$ convert at $39.56\%$ ($>45\%$ for $65+$), while prime working age (40–49) converts at only $7.92\%$. Median ages are identical (37 vs 38), but variance is $40\%$ higher for subscribers. | Stratified age KDE, life-stage cohort conversion table, and IQR outlier audit ($469$ records $>69$ years) (Section 3.1 Cells 19–21, Section 4.5 Cells 48–51). | Retain elderly records (do not trim as outliers); engineer demographic cohorts (`age_group`: `<30`, `30-39`, `40-49`, `50-59`, `60+`) in `BankFeatureEngineer` so linear models capture U-shape; standardize continuous age. |
| **EDA-08** | `education` | Non-monotonic response profile: `illiterate` ($22.22\%$) and `university.degree` ($13.72\%$) convert higher than intermediate basic tiers ($7.82\%-10.25\%$). `unknown` education converts at $14.50\%$. | Contingency analysis & Chi-square test ($\chi^2 = 193.11, p = 3.31 \times 10^{-38}$); ADR-002 benchmark (Section 3.1 Cell 21; ADR-002). | Reject arbitrary ordinal integer hierarchy (which incorrectly placed `unknown` below `illiterate` and forced monotonic linearity); One-Hot Encode all 8 tiers (`handle_unknown='ignore'`). |
| **EDA-09** | `job` | Extreme occupational divergence: students ($31.43\%$) and retirees ($25.23\%$) convert at $>2\times$ the baseline rate, while blue-collar ($6.89\%$), services ($8.14\%$), and entrepreneur ($8.52\%$) convert poorly. | Job frequency and response rate analysis ($N=875$ students, $N=1,720$ retired, $N=9,254$ blue-collar) (Section 3.1, Cells 19–21). | High-signal demographic predictor; encode as nominal One-Hot features; informs business segmentation targeting students and retirees for term deposits. |
| **EDA-10** | `contact` | Cellular communication achieves $14.74\%$ conversion ($N=26,144$) vs $5.23\%$ for fixed telephone ($N=15,044$)—a $2.82\times$ conversion superiority. | Communication channel frequency and bivariate response contrast (Section 3.2, Cells 23–25). | High-utility operational predictor; encode as binary/nominal; establish campaign routing policy prioritizing cellular contact channels. |
| **EDA-11** | `month` | Massive volume-efficiency mismatch: May accounts for $33.43\%$ of all calls ($13,769$) but has lowest conversion ($6.43\%$). Small campaigns in March ($50.55\%$), September ($44.91\%$), October ($43.87\%$), and December ($48.90\%$) show high yields. Jan/Feb are absent. | Monthly campaign volume vs response rate distribution; calendar cardinality audit (Section 3.2 Cells 23–25, Section 4.3 Cell 44). | One-Hot Encode month with `handle_unknown='ignore'`; use stratified random holdout to ensure September is represented in training data (ADR-001); recommend rebalancing call schedules. |
| **EDA-12** | `campaign` | Monotonic contact fatigue: conversion decays from $13.04\%$ on call 1 to $11.46\%$ on call 2, $9.39\%$ on call 4, $5.47\%$ on calls 7–10, and $3.11\%$ on $>10$ calls. Skewness $=4.76$ with maximum 56 calls. | Contact count conversion progression table; skewness & outlier audit (99th percentile $=14$) (Section 3.2 Cells 23–25, Section 4.5 Cells 48–51). | Scale using `RobustScaler` (median & IQR) rather than `StandardScaler` to handle heavy right tail; derive operational decision rule capping calls at 3 attempts per customer. |
| **EDA-13** | `previous` | Steep conversion escalation with prior campaign touches: 0 touches converts at $8.83\%$, 1 touch at $21.20\%$, 2 touches at $46.42\%$, and 3 touches at $59.26\%$. Heavy zero mass ($86.34\%$) and right skew ($3.83$). | Previous contact count distribution and conversion progression (Section 3.3 Cells 27–29, Section 4.5 Cells 48–51). | Apply `RobustScaler` to handle extreme zero-inflation and skewness; preserve numeric count feature alongside categorical outcome indicators. |
| **EDA-14** | `euribor3m`, `emp.var.rate`, `nr.employed` | Macroeconomic climate displays strongest linear correlation with target (`nr.employed` $r=-0.355$, `euribor3m` $r=-0.308$, `emp.var.rate` $r=-0.298$). Severe multicollinearity ($r > 0.90$ to $0.97$). Euribor $<1.28$ yields $\approx 50\%$ conversion vs $<4.5\%$ when Euribor $>4.8$. | Pearson correlation matrix heatmap; macroeconomic quintile response breakdown (Section 3.4, Cells 31–33). | Mandatory `StandardScaler` on all 5 economic variables to fix scale disparity (`nr.employed` $\sim 5000$ vs `euribor` $\sim 1-5$); mandatory L2/ElasticNet regularization for linear models to mitigate collinearity variance; confirms stratified split over chronological split (ADR-001). |
| **EDA-15** | `housing`, `loan` | Personal financial loans exhibit virtually zero relationship with deposit subscription: housing loan converts at $11.62\%$ (yes) vs $10.88\%$ (no); personal loan converts at $10.93\%$ (yes) vs $11.34\%$ (no) (baseline $11.27\%$). | Bivariate contingency tables and chi-square independence tests (Section 3.1, Cells 19–21). | Low predictive utility; encode nominally but expect near-zero feature importance or regularized regression coefficient shrinkage. |
| **EDA-16** | `day_of_week` | Day of week exhibits narrow, uniform response rates ($9.95\%$ Monday to $12.12\%$ Thursday). Weekend calls are completely absent ($0$ records). | Bivariate response distribution and format/cardinality audit (cardinality $=5$) (Section 3.2 Cells 23–25, Section 4.3 Cell 44). | Low-signal operational predictor; One-Hot Encode with regularization to prevent overfitting; safe to deprioritize during feature selection. |
| **EDA-17** | Full Dataset (`df`) | Exactly 12 duplicate rows ($0.029\%$, 24 rows total) identified across all 21 columns; exactly 12 duplicates excluding target `y`; zero conflicting labels across duplicate profiles (all 12 pairs are `no`). | Exact row-match and feature-subset duplicate diagnostics (Section 4.1, Cells 36–38). | Prune duplicate rows prior to train/test splitting (`df.drop_duplicates()`) to prevent identical profiles leaking across validation folds and overweighting the objective loss function. |
| **EDA-18** | All 10 Numerical Features | All numerical variables strictly satisfy domain boundaries with 0 negative values where physical counts are expected (`cons.conf.idx` correctly negative $[-50.8, -26.9]$). Severe scale disparity exists across variables ($0.63$ to $5,228$). | Min/max domain audit against physical boundaries (Section 4.2, Cells 39–41). | Apply `StandardScaler` to macroeconomic variables and continuous age; apply `RobustScaler` to skewed counts (`campaign`, `previous`) to prevent gradient and distance distortion. |
| **EDA-19** | `default='yes'`, `illiterate`, `dec` | Highly rare category levels: `default='yes'` ($N=3$, $0.007\%$, 0 subscriptions), `illiterate` ($N=18$, $0.044\%$, 4 subscriptions), `month='dec'` ($N=182$, $0.442\%$, 89 subscriptions). | Prevalence scan below $1.0\%$ threshold (Section 4.4, Cells 45–47). | Enforce L2 regularization to prevent infinite weights in logistic regression (quasi-complete separation for `default='yes'`); configure `handle_unknown='ignore'` to handle rare categories in CV folds. |
| **EDA-20** | Full Dataset (Temporal vs Random Partitioning) | Chronological ordering exhibits extreme 2008 financial crisis regime shift: Euribor drops from $4.27$ to $1.03$ with zero distribution overlap; positive rate surges by $384\%$ ($6.37\%$ to $30.83\%$); September has 0 training records. | Chronological 80/20 vs Stratified 80/20 empirical comparative benchmark; ADR-001 (Section 3.4 Cell 33; ADR-001). | Formally abandon chronological holdout; adopt Stratified Random Holdout (80/20, `random_state=42`) with Stratified 5-Fold CV to maintain valid priors ($11.27\%$), full covariate support, and stable evaluation. |

---

## 3. Detailed Thematic Breakdowns

### 3.1 Target Distribution & Evaluation Architecture (`EDA-01`)
- **Empirical Context**: Out of 41,188 total records, 36,548 customers ($88.73\%$) declined the term deposit (`y = 'no'`), while only 4,640 customers ($11.27\%$) subscribed (`y = 'yes'`).
- **Core Implication**: A naive majority-class baseline achieves an $88.73\%$ accuracy while identifying 0 positive prospects. Model performance must be evaluated using Precision-Recall Area Under Curve (PR-AUC), Receiver Operating Characteristic (ROC-AUC), and threshold-tuned F1/lift metrics. Stratification across development and holdout splits is mandatory to preserve the $11.27\%$ prior.

### 3.2 Data Leakage Quarantine (`EDA-04`)
- **Empirical Context**: Call `duration` exhibits the highest linear correlation with subscription ($r = +0.405$). The median call duration for subscribers ($449.0$s) is nearly triple that of non-subscribers ($163.5$s). Calls lasting longer than 10 minutes achieve a $48.61\%$ conversion rate, whereas calls under 1 minute yield only $0.02\%$.
- **Core Implication**: Call duration is unobservable before an outbound call is dialed. Including `duration` creates catastrophic post-event data leakage. The feature is quarantined and excluded entirely from the predictive feature pipeline (`remainder='drop'`).

### 3.3 Historical Campaign Interaction & Sentinel Semantics (`EDA-02`, `EDA-05`, `EDA-06`, `EDA-13`)
- **Empirical Context**: 
  - Past campaign success (`poutcome = 'success'`) yields a $65.11\%$ conversion rate ($5.8\times$ baseline).
  - $96.32\%$ of records ($N = 39,673$) carry the sentinel value `pdays = 999` (never contacted previously in past campaigns), converting at $9.26\%$.
  - In contrast, genuine elapsed day counts ($[0, 27]$ days, $N = 1,515$) convert at $63.83\%$.
  - **Critical Anomaly**: $4,110$ records have `previous > 0` but `pdays = 999`. All $4,110$ records have `poutcome = 'failure'`. The legacy CRM defaulted elapsed days to 999 for older failed attempts while logging non-zero `previous` contact counts.
- **Core Implication**:
  - Treating `pdays = 999` continuously imposes arbitrary Euclidean distances ($999$ vs $6$ days).
  - Deriving a contact indicator solely as `pdays != 999` misclassifies $4,110$ warm leads as virgin leads.
  - Transformation strategy: Derive binary `previously_contacted = (pdays != 999).astype(int)`, discretize valid days into `pdays_group`, drop raw `pdays`, and preserve `previous` and `poutcome` in the feature pipeline.

### 3.4 Demographic Non-Linearities & Educational Representation (`EDA-03`, `EDA-07`, `EDA-08`, `EDA-09`, `EDA-15`)
- **Empirical Context**:
  - **Age**: Displays a distinct U-shaped response curve. Youth ($<25$, $23.97\%$) and seniors ($60+$, $39.56\%$) convert at elevated rates, while working adults ($40-49$) convert at only $7.92\%$.
  - **Education**: Displays non-monotonic response (`illiterate` $22.22\%$, `university.degree` $13.72\%$, `basic.9y` $7.82\%$). `unknown` education converts at $14.50\%$.
  - **Credit Default**: `default = 'yes'` has only 3 observations ($0$ subscribers); `default = 'unknown'` has $8,597$ observations ($20.87\%$) converting at $5.15\%$ vs $12.88\%$ for clean credit.
  - **Liabilities**: Housing and personal loans convert at $11.62\%$ and $10.93\%$, virtually indistinguishable from baseline ($11.27\%$).
- **Core Implication**:
  - Naive outlier removal based on IQR bounds for age would discard seniors ($N = 1,020$), who represent the bank's highest-converting demographic.
  - Categorical binning of `age` into `age_group` cohorts (`<30`, `30-39`, `40-49`, `50-59`, `60+`) enables linear classifiers to capture non-linear extremes.
  - Rejection of ordinal encoding for `education` (ADR-002) in favor of One-Hot Encoding avoids penalizing `unknown` and respects non-monotonicity.
  - `default` operates as a missingness indicator; L2 regularization prevents weight divergence.

### 3.5 Operational Campaign Dynamics & Contact Fatigue (`EDA-10`, `EDA-11`, `EDA-12`, `EDA-16`)
- **Empirical Context**:
  - **Channel**: Cellular outreach ($14.74\%$) achieves a $2.82\times$ conversion advantage over fixed landlines ($5.23\%$).
  - **Month**: May accounts for $33.43\%$ of all calls ($13,769$) but delivers the worst conversion rate ($6.43\%$). Selective campaigns in March ($50.55\%$), September ($44.91\%$), October ($43.87\%$), and December ($48.90\%$) show high yields. January and February are unobserved ($0$ records).
  - **Contact Fatigue**: Conversion decays monotonically from $13.04\%$ on call 1 down to $3.11\%$ on $>10$ calls. Skewness is $4.76$ (up to 56 calls).
- **Core Implication**:
  - `campaign` count must be scaled via `RobustScaler` (median & IQR) rather than `StandardScaler` to prevent leverage distortion from extreme contact outliers.
  - All nominal operational attributes (`contact`, `month`, `day_of_week`) are One-Hot Encoded with `handle_unknown='ignore'`.
  - Insights motivate an operational business rule recommending termination of outbound calling after 3 failed attempts.

### 3.6 Macroeconomic Regime Shift & Validation Strategy (`EDA-14`, `EDA-17`, `EDA-18`, `EDA-19`, `EDA-20`)
- **Empirical Context**:
  - Macroeconomic indicators are the strongest linear predictors (`nr.employed` $r = -0.355$, `euribor3m` $r = -0.308$, `emp.var.rate` $r = -0.298$).
  - Severe collinearity exists among indicators ($\text{corr}(\text{euribor3m}, \text{emp.var.rate}) = 0.972$).
  - Chronological 80/20 partitioning places the 2008–2010 crisis regime shift into the test partition: `euribor3m` ranges $[1.3, 5.0]$ in training vs $[0.6, 1.3]$ in testing with zero overlap; positive base rate surges from $6.37\%$ to $30.83\%$.
- **Core Implication**:
  - Scale disparities (`nr.employed` $\sim 5000$ vs `euribor3m` $\sim 1-5$) mandate `StandardScaler` across all 5 macro features.
  - Linear models require L2 regularization to control variance inflation caused by collinearity.
  - Chronological holdout was formally abandoned in ADR-001 in favor of Stratified Random Holdout (80/20, `random_state=42`) with 5-Fold Stratified Cross-Validation to guarantee equal target balance ($11.27\%$) and shared covariate support.

---

## 4. Traceability Matrix: EDA Insights to Architecture & Code

This matrix demonstrates end-to-end traceability from each EDA finding to its concrete implementation in the codebase and documentation.

| EDA ID | Notebook Section & Evidence Cells | Preprocessing Implementation | Decision Log Governance |
| :--- | :--- | :--- | :--- |
| **EDA-01** | Section 2 (Cells 13–15) | Stratified 80/20 holdout in [`02_Data_Preprocessing_Pipeline.ipynb`](file:///Users/moni/Desktop/ML%20proj/notebooks/02_Data_Preprocessing_Pipeline.ipynb#L22) | ADR-001, ADR-003 |
| **EDA-02** | Section 3.3 (Cells 27–29) | `OneHotEncoder` on `poutcome` in `ColumnTransformer` | ADR-003, ADR-004 |
| **EDA-03** | Section 1.2 (Cell 10), 4.4 (Cell 47) | `OneHotEncoder` retaining `default_unknown` as indicator | ADR-003, ADR-004 |
| **EDA-04** | Section 4.8 (Cells 59–62) | Quarantined via `remainder='drop'` in `ColumnTransformer` | ADR-003 |
| **EDA-05** | Section 4.6 (Cells 52–55) | `BankFeatureEngineer` (`previously_contacted`, `pdays_group`) in [`src/transformers.py`](file:///Users/moni/Desktop/ML%20proj/src/transformers.py#L26-L36) | ADR-003, ADR-004 |
| **EDA-06** | Section 4.7 (Cells 56–58) | Preservation of `previous` and `poutcome` alongside binned `pdays` | ADR-003 |
| **EDA-07** | Section 3.1 (Cells 19–21), 4.5 (Cell 51) | `BankFeatureEngineer` (`age_group`), `StandardScaler` on continuous age | ADR-003, ADR-004 |
| **EDA-08** | Section 3.1 (Cell 21) | `OneHotEncoder` for `education` (71-feature schema) | ADR-002, ADR-004 |
| **EDA-09** | Section 3.1 (Cells 19–21) | `OneHotEncoder` on `job` with `handle_unknown='ignore'` | ADR-004 |
| **EDA-10** | Section 3.2 (Cells 23–25) | `OneHotEncoder` on `contact` | ADR-004 |
| **EDA-11** | Section 3.2 (Cells 23–25), 4.3 (Cell 44) | `OneHotEncoder` on `month` (`handle_unknown='ignore'`) | ADR-001, ADR-004 |
| **EDA-12** | Section 3.2 (Cells 23–25), 4.5 (Cell 51) | `RobustScaler` on `campaign` in `ColumnTransformer` | ADR-003, ADR-004 |
| **EDA-13** | Section 3.3 (Cells 27–29), 4.5 (Cell 51) | `RobustScaler` on `previous` in `ColumnTransformer` | ADR-003, ADR-004 |
| **EDA-14** | Section 3.4 (Cells 31–33) | `StandardScaler` on macroeconomic variables; L2 regularization | ADR-001, ADR-003, ADR-004 |
| **EDA-15** | Section 3.1 (Cells 19–21) | `OneHotEncoder` on `housing` and `loan` | ADR-004 |
| **EDA-16** | Section 3.2 (Cells 23–25), 4.3 (Cell 44) | `OneHotEncoder` on `day_of_week` | ADR-004 |
| **EDA-17** | Section 4.1 (Cells 36–38) | Deduplication (`df.drop_duplicates()`) prior to partitioning | ADR-003 |
| **EDA-18** | Section 4.2 (Cells 39–41) | Two-tier scaling architecture (`StandardScaler` + `RobustScaler`) | ADR-003, ADR-004 |
| **EDA-19** | Section 4.4 (Cells 45–47) | L2 regularization and `handle_unknown='ignore'` | ADR-003 |
| **EDA-20** | Section 3.4 (Cell 33) | Stratified 80/20 holdout with Stratified 5-Fold Cross-Validation | ADR-001 |

---

## 5. Report Reusability Kit: Synthesis for Final Submission

These synthesis statements and summaries are pre-formatted for direct inclusion in the **Problem Framing & Exploratory Data Analysis** section of the final project report:

1. **Target Imbalance & Evaluation Framing**:
   > *"The empirical subscription base rate is $11.27\%$ ($4,640$ subscribers out of $41,188$ contacts). Evaluating candidate models purely on classification accuracy would yield a deceptive $88.7\%$ performance for a null predictor. Model selection is therefore anchored on PR-AUC, ROC-AUC, and expected business utility across decile lead rankings (EDA-01)."*

2. **Data Leakage Quarantine (`duration`)**:
   > *"Although call duration exhibits strong linear correlation with subscription ($r = +0.405$), it is physically unobservable at the moment of outbound prospect scoring. Incorporating duration introduces post-event target leakage that collapses operational utility. In accordance with strict leakage governance, duration was excluded prior to feature engineering (EDA-04)."*

3. **Domain Feature Engineering (`pdays` & `age`)**:
   > *"Exploratory audit revealed that $96.32\%$ of records contain the sentinel value `pdays = 999`, creating an artificial metric distortion if treated continuously. A custom transformer (`BankFeatureEngineer`) derived a binary interaction flag (`previously_contacted`) and recency cohorts (`pdays_group`), while binning age into demographic life-stage cohorts (`age_group`) to capture the empirical U-shaped response curve among youth ($23.97\%$) and seniors ($39.56\%$) (EDA-05, EDA-07)."*

4. **Macroeconomic Climate & Collinearity Management**:
   > *"Macroeconomic indicators (`euribor3m`, `emp.var.rate`, `nr.employed`) exhibit the strongest linear correlation with subscription ($r \approx -0.30$ to $-0.35$), reflecting a flight-to-safety dynamic during the 2008–2010 financial crisis. Due to extreme multicollinearity ($r > 0.95$), linear candidate architectures enforce L2 shrinkage, while tree ensembles naturally partition the correlated feature space (EDA-14)."*
