# Exploratory Data Analysis (EDA) - Text & Comments

> Extracted text blocks (markdown cells) and code comments from [`notebooks/01_Problem_Framing_EDA.ipynb`](file:///Users/moni/Desktop/ML proj/notebooks/01_Problem_Framing_EDA.ipynb).

---

> **Code Comments (Cell 0):**
> - Deterministically resolve project root independent of caller working directory
> - Locate dataset dynamically with fallback candidate paths

### Dataset Overview

The Bank Marketing dataset contains 41,188 customer campaign records and 21 attributes. Each record represents a customer contact during a marketing campaign.

The dataset includes customer demographic information, financial information, campaign-related details, and economic indicators. The target variable is `y`, which represents whether the customer subscribed to the term deposit (`yes` or `no`).

The dataset contains both numerical and categorical variables. No missing/null values were identified during the initial data quality check.

### Unknown Value Analysis

Although the dataset does not contain traditional missing values (null values), several categorical variables contain the value "unknown", which represents unavailable customer information.

The distribution of unknown values was analyzed to understand data quality issues.

The highest number of unknown values appears in the `default` feature (8,597 records), followed by `education` (1,731 records). Other variables such as `housing`, `loan`, `job`, and `marital` contain fewer unknown values.

These unknown values should not be removed immediately because they may contain useful information. The handling strategy will be decided during the preprocessing stage based on their impact on model performance.

## Target Variable Analysis

The target variable `y` represents whether a customer subscribed to the bank's term deposit after the marketing campaign.

- `yes` indicates successful subscription.
- `no` indicates that the customer did not subscribe.

The analysis shows that out of 41,188 customer records, 36,548 customers did not subscribe, while only 4,640 customers subscribed.

The target variable is highly imbalanced, with 88.73% of records belonging to the `no` class and only 11.27% belonging to the `yes` class.

This class imbalance is an important consideration for model development because a model may achieve high accuracy by mainly predicting the majority class. Therefore, evaluation should consider additional metrics such as precision, recall, F1-score, ROC-AUC, and PR-AUC.

Numerical Feature Summary
The descriptive statistics provide an overview of the numerical variables in the Bank Marketing dataset, including their distribution, central tendency, and range.

The dataset contains 41,188 records, and all numerical variables have complete observations.

Key observations:

The average customer age is approximately 40 years, with ages ranging from 17 to 98 years.
The duration feature has a wide range, from 0 to 4918 seconds, indicating high variation in call durations and possible outliers.
The campaign feature shows that customers were contacted an average of approximately 2–3 times during the current campaign, although some customers received many contacts.
The pdays feature contains many values of 999, which represents customers who were not previously contacted.
The previous feature indicates that most customers had limited previous campaign interactions.
Economic variables such as emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, and nr.employed provide information about the economic conditions during the campaign period.

---

# 3. Exploratory Data Analysis by Feature Family & Target Relationships

To establish an evidence-based foundation for subsequent preprocessing pipelines and model architectures, exploratory data analysis must move beyond disconnected univariate distributions. In telemarketing response prediction, predictors operate within distinct operational, demographic, and macro-financial contexts.

In accordance with project guidelines, this section systematically analyzes the four major predictor families:
1. **Customer Profile Variables**: Demographics (`age`, `job`, `marital`, `education`) and financial commitments (`default`, `housing`, `loan`).
2. **Current Campaign Operational Variables**: Communication channel (`contact`), temporal placement (`month`, `day_of_week`), and contact intensity (`campaign`).
3. **Historical Campaign Context Variables**: Prior contact volume (`previous`), recency (`pdays`), and past campaign outcome (`poutcome`).
4. **Macroeconomic Climate Indicators**: Employment variation (`emp.var.rate`), price indices (`cons.price.idx`, `cons.conf.idx`), benchmark interest rate (`euribor3m`), and labor force volume (`nr.employed`).

To avoid redundant, low-value visual plots, analyses are grouped into multi-panel comparative figures. Each feature family addresses four fundamental analytical questions:
- **What pattern exists?** (Underlying distributions, frequencies, and spread)
- **Does it relate to the target?** (Response rate contrasts against the 11.27% dataset baseline)
- **Why does it matter?** (Behavioral, financial, or operational domain mechanisms)
- **Does it affect preprocessing/feature engineering/model evaluation?** (Direct architectural implications)

### 3.1 Customer Profile Variables (Demographics & Financial Standing)

Customer profile variables capture the personal life stage, socio-economic stratum, and debt obligations of prospective clients. We examine whether personal attributes govern a customer's propensity to lock funds into a long-term savings instrument.

> **Code Comments (Cell 19):**
> - 1. Stratified Quantitative Analysis of Age by Campaign Response
> - 2. Conversion Rate across Life-Stage Age Cohorts
> - 3. Categorical Customer Variables Response Rates

> **Code Comments (Cell 20):**
> - High-Signal Comparative Visualizations: Customer Profile Variables
> - 1. Age Distribution by Target (KDE & Median Contrast)
> - 2. Conversion Rate across Life-Stage Age Cohorts
> - 3. Response Rate by Occupation (job)
> - 4. Response Rate by Marital Status
> - 5. Education Attainment Response Profile
> - 6. Financial Obligations Contrast (default, housing, loan)

#### Findings & Modelling / Preprocessing Implications: Customer Profile Variables

| Dimension | Analytical Finding & Evidence |
| :--- | :--- |
| **What pattern exists?** | 1. **Age Bimodal U-Shape**: While overall central tendencies are deceptively identical (median age $38.0$ for non-subscribers vs $37.0$ for subscribers), the variance for subscribers is $40\%$ wider ($\sigma=13.84$ vs $9.90$). Clients under 25 achieve a **$23.97\%$** conversion rate, and seniors aged 60+ achieve a **$39.56\%$** conversion rate. Conversely, prime working-age individuals (40–49) hit a trough of only **$7.92\%$**.<br>2. **Occupation Disparities**: Non-labor-force participants demonstrate the highest conversion: `student` ($31.43\%$) and `retired` ($25.23\%$). Salaried professionals convert near baseline (`admin.`: $12.97\%$, `technician`: $10.82\%$, `management`: $11.22\%$), whereas `blue-collar` workers convert at only $6.89\%$.<br>3. **Marital Disparities**: `single` clients convert at **$14.00\%$**, significantly outperforming `married` ($10.16\%$) and `divorced` ($10.32\%$) individuals.<br>4. **Education Non-Monotonicity**: Attainment exhibits a distinctly non-monotonic conversion profile: `illiterate` ($22.22\%$) and `university.degree` ($13.72\%$) outperform intermediate tiers (`basic.9y` at $7.82\%$ and `basic.6y` at $8.20\%$). Clients with `unknown` education convert at $14.50\%$.<br>5. **Loan Independence & Default Sparsity**: `housing` loans ($11.62\%$ yes vs $10.88\%$ no) and `loan` personal liabilities ($10.93\%$ yes vs $11.34\%$ no) exhibit near-identical response rates close to baseline ($11.27\%$). `default='yes'` possesses only 3 records (zero conversions), while `default='unknown'` converts at only $5.15\%$ vs $12.88\%$ for `default='no'`. |
| **Does it relate to the target?** | Strong, statistically significant non-linear relationships exist for `age`, `job`, `marital`, and `education`. In contrast, `housing` and `loan` exhibit near-zero marginal association with the target. `default` acts not as a tri-level credit risk metric, but as an information-availability proxy (verified clean credit vs missing records). |
| **Why does it matter?** | **Financial Psychology of Term Deposits**: A term deposit is a low-risk, illiquid savings vehicle requiring capital lockup for a fixed duration. Demographic segments with discretionary liquid savings and low short-term financial commitments—such as retirees seeking capital preservation and students/unmarried individuals with minimal liabilities—naturally exhibit higher demand. In contrast, mid-career family heads (ages 35–50, married, with dependents) have higher routine consumption commitments and lower unallocated cash balances. Existing retail debt (`housing` or `personal loan`) does not suppress term deposit appetite because borrowers and savers frequently coexist within the same retail banking customer segments. |
| **Does it affect preprocessing/feature engineering/model evaluation?** | 1. **Feature Engineering**: Validates the creation of demographic life-stage cohorts (`age_group`: `<30`, `30-39`, `40-49`, `50-59`, `60+`) in `src.transformers.BankFeatureEngineer`. Categorical binning enables linear models (Logistic Regression) to assign distinct positive coefficients to youth and retirement tiers without being constrained by a single linear slope.<br>2. **Categorical Encoding**: Reaffirms **ADR-002**: `education` must be One-Hot Encoded rather than mapped to an ordinal integer hierarchy ($0$ to $7$). Forcing ordinal ranking incorrectly penalizes `illiterate` and `unknown` categories and imposes invalid equal spacing. All nominal variables (`job`, `marital`, `default`, `housing`, `loan`) require One-Hot Encoding with `handle_unknown='ignore'`.<br>3. **Scaling**: Continuous `age` should be standardized using `StandardScaler` to maintain numerical stability during gradient-based optimization.<br>4. **Modelling & Evaluation**: Linear models will require the binned `age_group` features to capture age non-linearities, whereas tree ensembles (Random Forest, Gradient Boosting) can partition continuous age directly. |

> **Governance Alignment**: Formally ratified under **[ADR-002](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-002)** (One-Hot Encoding for `education`) and **[ADR-004](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-004)** (Customer Profile Family Rationales & PRE-09 Dual Age Representation retaining both continuous `age` and categorical `age_group`).

### 3.2 Current Campaign Operational Variables

Current campaign variables (`contact`, `month`, `day_of_week`, `campaign`) reflect the telemarketing campaign's execution parameters—specifically how, when, and how frequently the bank reached out to each client during the active campaign.

> **Code Comments (Cell 23):**
> - 1. Quantitative Analysis of Communication Channel (contact)
> - 2. Monthly Campaign Dynamics (Volume vs Conversion)
> - 3. Day of Week Conversion Rates
> - 4. Contact Fatigue Analysis (campaign call frequency)

> **Code Comments (Cell 24):**
> - High-Signal Comparative Visualizations: Campaign Operational Variables
> - 1. Communication Channel Conversion
> - 2. Monthly Campaign Dynamics: Call Volume vs Conversion Rate (Dual Axis)
> - Combine legends for dual axis
> - 3. Day of Week Conversion Rates
> - 4. Campaign Contact Fatigue Curve

#### Findings & Modelling / Preprocessing Implications: Campaign Operational Variables

| Dimension | Analytical Finding & Evidence |
| :--- | :--- |
| **What pattern exists?** | 1. **Channel Disparity**: Cellular outreach achieves a **$14.74\%$** conversion rate ($N=26,144$) vs only **$5.23\%$** for landline telephone ($N=15,044$)—a **$2.8\times$ conversion superiority**.<br>2. **Severe Monthly Operational Imbalance**: Outbound calling displays extreme volume and efficiency decoupling. May accounts for a massive **$33.43\%$** of all contacts ($13,769$ calls) but yields the worst conversion rate in the dataset (**$6.43\%$**). In contrast, March ($50.55\%$), September ($44.91\%$), October ($43.87\%$), and December ($48.90\%$) exhibit stellar conversion rates on selective, low-volume campaigns ($<800$ calls each).<br>3. **Day-of-Week Uniformity**: Conversion rates across business days remain tightly bounded between $9.95\%$ (Monday) and $12.12\%$ (Thursday), exhibiting minimal variance.<br>4. **Monotonic Contact Fatigue**: Success probability strictly degrades as calling frequency increases: $1^{\text{st}}$ contact ($13.04\%$), $2^{\text{nd}}$ ($11.46\%$), $3^{\text{rd}}$ ($10.75\%$), $4^{\text{th}}$ ($9.39\%$), $5-6$ contacts ($7.56\%$), $7-10$ contacts ($5.47\%$), and $>10$ contacts ($3.11\%$). |
| **Does it relate to the target?** | `contact`, `month`, and `campaign` contact frequency relate strongly and significantly to campaign response. `day_of_week` has negligible predictive utility and minimal effect size. |
| **Why does it matter?** | 1. **Operational Waste & Call Fatigue**: Contacting an unresponsive prospect more than 3 times generates steep diminishing returns and customer irritation. Over $5,000$ calls were placed to individuals contacted $4+$ times, producing sub-par yields that squander call center capacity.<br>2. **Channel Reachability**: Cellular phones reach prospective clients directly and represent personal communication devices. Fixed landlines are disproportionately answered by gatekeepers or unassisted households.<br>3. **Quarter-End / Tax Seasonality**: High-performing months align with financial milestones (March tax planning, September back-to-work, December year-end bonus allocations). |
| **Does it affect preprocessing/feature engineering/model evaluation?** | 1. **Scaling Choice for `campaign`**: The `campaign` feature exhibits extreme positive skewness ($4.76$) with outliers reaching up to $56$ contacts. Applying `StandardScaler` would be corrupted by extreme outliers; this directly justifies applying `RobustScaler` (median & IQR centering) to `campaign` in the preprocessing pipeline.<br>2. **Categorical Encoding**: `contact`, `month`, and `day_of_week` are nominal and should be One-Hot Encoded with `handle_unknown='ignore'`.<br>3. **Holdout Split Reassurance (ADR-001)**: September is exclusively concentrated between row indices $37,887$ and $40,855$. Chronological partitioning completely starves the training set of `month='sep'`, creating an out-of-vocabulary test anomaly. Stratified random holdout guarantees uniform category coverage across all 10 months.<br>4. **Modelling & Policy**: Model decile lift analysis should be utilized to establish operational call-termination rules (e.g., stopping outreach after 3 attempts if predicted response probability remains low). |

> **Governance Alignment**: Formally ratified under **[ADR-004](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-004)** (Current Campaign Operational Family Rationales: cellular channel superiority, monthly volume rebalancing, and contact fatigue scaling via `RobustScaler`).

### 3.3 Historical Campaign Context Variables

Historical campaign variables (`previous`, `pdays`, `poutcome`) record the bank's prior outreach history with each client before the current campaign began. They provide critical insight into customer relationship warmth and long-term bank loyalty.

> **Code Comments (Cell 27):**
> - 1. Conversion Rate Progression by Previous Contact Count (previous)
> - 2. Conversion Contrast: pdays Sentinel (999) vs Elapsed Contact Recency Cohorts
> - 3. Historical Campaign Outcome (poutcome) Performance

> **Code Comments (Cell 28):**
> - High-Signal Comparative Visualizations: Historical Campaign Context Variables
> - 1. Prior Contact Count (previous)
> - 2. pdays Contact Recency Cohorts
> - 3. Previous Campaign Outcome (poutcome)

#### Findings & Modelling / Preprocessing Implications: Historical Campaign Context Variables

| Dimension | Analytical Finding & Evidence |
| :--- | :--- |
| **What pattern exists?** | 1. **Prior Contact Lift**: First-time outreach prospects (`previous = 0`, $N=35,563$, $86.34\%$) convert at only **$8.83\%$**. With just 1 prior interaction, response jumps to **$21.20\%$**. At 2 prior interactions, conversion reaches **$46.42\%$**; at 3 interactions, **$59.26\%$**.<br>2. **Recency Decoupling**: Clients with genuine elapsed contact days (`pdays < 999`, $N=1,515$, $3.68\%$) achieve an outstanding conversion rate of **$63.83\%$**. Recency within 0–6 days achieves $\approx 65-66\%$ conversion. Conversely, clients uncontacted in prior campaigns (`pdays = 999`, $N=39,673$) convert at only **$9.26\%$**.<br>3. **Previous Success as Prime Signal**: A past successful campaign response (`poutcome = 'success'`, $N=1,373$, $3.33\%$) converts at an astonishing **$65.11\%$**—a **$5.8\times$ lift over baseline**. Even previously failed contacts (`poutcome = 'failure'`, $N=4,252$) convert at $14.23\%$, outperforming virgin cold leads (`nonexistent` at $8.83\%$). |
| **Does it relate to the target?** | Historical campaign interaction is the **single most predictive feature family in the entire dataset**. A client's past interaction history with the bank dominates static demographic characteristics in predicting subscription propensity. |
| **Why does it matter?** | **Lead Warmth & Trust Equilibrium**: In banking sales, warm leads represent established relationships where product fit, identity verification, and trust have already been established. Repeat subscribers demonstrate proven interest in fixed-yield savings. Even a failed past call establishes brand recognition, making the customer more receptive than an uncontacted cold prospect. |
| **Does it affect preprocessing/feature engineering/model evaluation?** | 1. **Feature Engineering**: Directly mandates the feature engineering implemented in `src.transformers.BankFeatureEngineer`: raw `pdays` cannot be supplied as a continuous metric variable due to the sentinel value $999$ ($96.32\%$ of dataset). It is transformed into:<br>   - `previously_contacted`: binary indicator ($1$ if `pdays != 999` else $0$);<br>   - `pdays_group`: categorical recency cohorts (`0_to_6_days`, `7_to_14_days`, `15_plus_days`, `not_contacted`);<br>   - dropping raw `pdays` to prevent invalid Euclidean distance distortions.<br>2. **Scaling Choice for `previous`**: With $86.34\%$ zero entries and a heavy right tail (skewness $3.83$), `previous` must be scaled using `RobustScaler` rather than `StandardScaler`.<br>3. **Modelling & Subgroup Diagnostics**: Models will naturally assign dominant weights to `poutcome_success` and `previously_contacted`. However, because $86.3\%$ of prospects are completely cold leads (`previous=0`), model evaluation must evaluate discriminative performance (ROC-AUC / PR-AUC) on the cold-lead sub-population to ensure the model does not degenerate into a trivial lookup table for repeat customers. |

> **Governance Alignment**: Formally ratified under **[ADR-003](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-003)** (Historical Campaign Feature Preservation) and **[ADR-004](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-004)** (Historical Campaign Context Family Rationales: preserving `previous` and `poutcome` alongside engineered `pdays_group`).

### 3.4 Macroeconomic Context & Indicator Multicollinearity

The dataset includes 5 macroeconomic indicators captured at monthly or quarterly frequency:
- `emp.var.rate`: Employment variation rate (quarterly indicator);
- `cons.price.idx`: Consumer price index (monthly indicator);
- `cons.conf.idx`: Consumer confidence index (monthly indicator);
- `euribor3m`: Euribor 3-month interest rate (daily indicator, recorded at contact);
- `nr.employed`: Number of employees in the economy (quarterly indicator, in thousands).

These features reflect external macroeconomic climate during the 2008–2010 financial crisis era.

> **Code Comments (Cell 31):**
> - 1. Stratified Descriptive Statistics for Macroeconomic Variables by Target
> - 2. Pearson Correlation Matrix: Economic Indicators and Target Response
> - 3. Extreme Multicollinearity Check (|r| >= 0.70)

> **Code Comments (Cell 32):**
> - High-Signal Comparative Visualizations: Macroeconomic Climate & Collinearity
> - 1. Annotated Correlation Heatmap
> - 2. Standardized Economic Feature Distributions by Target (Boxplots)
> - 3. Macro Regime Shift Scatter: euribor3m vs emp.var.rate (N=5,000 Sample)
> - 4. Conversion Rate across Euribor 3M Quantiles (Regime Cliff)

#### Findings & Modelling / Preprocessing Implications: Macroeconomic Variables

| Dimension | Analytical Finding & Evidence |
| :--- | :--- |
| **What pattern exists?** | 1. **Inverse Macroeconomic Association**: All primary economic indicators exhibit substantial negative correlations with campaign subscription:<br>   - `nr.employed`: Pearson $r = -0.355$ (median $5099.1$ for subscribers vs $5195.8$ for non-subscribers);<br>   - `euribor3m`: Pearson $r = -0.308$ (median $1.266$ for subscribers vs $4.857$ for non-subscribers);<br>   - `emp.var.rate`: Pearson $r = -0.298$ (median $-1.8$ for subscribers vs $+1.1$ for non-subscribers);<br>   - `cons.price.idx`: Pearson $r = -0.136$ (mean $93.35$ for subscribers vs $93.60$ for non-subscribers);<br>   - `cons.conf.idx`: Pearson $r = +0.055$ (subtle positive association).<br>2. **Severe Multicollinearity**: The economic variables are extraordinarily collinear:<br>   - $\text{corr}(\text{euribor3m}, \text{emp.var.rate}) = +0.972$<br>   - $\text{corr}(\text{euribor3m}, \text{nr.employed}) = +0.945$<br>   - $\text{corr}(\text{emp.var.rate}, \text{nr.employed}) = +0.907$<br>   - $\text{corr}(\text{emp.var.rate}, \text{cons.price.idx}) = +0.775$<br>3. **Regime Cliff**: When `euribor3m` is in the lowest quintile ($[0.634, 1.281]$), subscription rate surges to $\approx 50\%$. When `euribor3m` exceeds $4.8$, conversion collapses to under $4.5\%$. |
| **Does it relate to the target?** | Yes. Macroeconomic indicators provide the **highest linear correlation with campaign subscription** among all numerical features in the dataset. External monetary conditions serve as a massive macro-level tide lifting or sinking response rates. |
| **Why does it matter?** | **Flight-to-Safety Macroeconomics (2008–2010 Financial Crisis)**: The dataset was gathered between May 2008 and November 2010, capturing the global financial crisis and European debt crisis. When macroeconomic indicators collapsed (negative employment variation, ECB interest rate cuts to historic lows), economic uncertainty triggered widespread risk aversion. Retail banking clients sought safe-haven capital protection in fixed-rate, principal-guaranteed bank term deposits. Conversely, during high-rate, pre-crisis periods (early 2008), equity markets and alternative investments absorbed capital, suppressing term deposit appeal. |
| **Does it affect preprocessing/feature engineering/model evaluation?** | 1. **Feature Scaling (Mandatory `StandardScaler`)**: The 5 economic indicators operate across vastly discordant numerical scales (`nr.employed` $\sim 5000$, `cons.price.idx` $\sim 93$, `euribor3m` $\sim 1-5$, `emp.var.rate` $\sim -3$ to $+1$, `cons.conf.idx` $\sim -50$ to $-26$). Without standardization, gradient-based estimators (Logistic Regression, Neural Networks) and distance-based estimators will be completely dominated by `nr.employed` purely due to metric magnitude.<br>2. **Multicollinearity in Linear Models**: In unregularized linear models (standard OLS or unpenalized Logistic Regression), extreme correlation ($r > 0.95$) inflates parameter variance, yields erratic coefficients, and causes sign reversals. Linear models **must enforce L2 regularization (Ridge penalty)** or ElasticNet.<br>3. **Tree Ensemble Robustness & Feature Importance Dilution**: Decision trees and ensembles (Random Forest, XGBoost, LightGBM) are naturally invariant to monotonic feature scaling and collinearity. However, feature importance metrics (MDI / Gini importance) will split credit across the collinear trio (`euribor3m`, `emp.var.rate`, `nr.employed`), diluting their individual importance ranks.<br>4. **Validation Split Governance (ADR-001 Validation)**: This empirical regime shift provides the decisive justification for ADR-001. Under chronological holdout, the entire low-interest-rate regime ($[0.634, 1.299]$) fell into the test set with zero overlap in the development set, guaranteeing out-of-distribution model collapse. Stratified random holdout ensures both development and test partitions span the complete macroeconomic cycle. |

> **Governance Alignment**: Formally ratified under **[ADR-001](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-001)** (Stratified Holdout over Temporal Partitioning) and **[ADR-004](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-004)** (Macroeconomic Indicator Family Rationales: `StandardScaler` on all 5 indicators and L2/Ridge shrinkage to control severe collinearity).

### 3.5 Synthesis of EDA Insights & Downstream Preprocessing/Modelling Blueprint

The exploratory findings across all four predictor families are synthesized into an integrated architectural blueprint connecting observed data patterns directly to data engineering and model evaluation decisions:

| Predictor Family | Features Covered | Observed Pattern & Target Association | Domain Mechanism | Preprocessing & Feature Engineering Choice | Downstream Modelling & Evaluation Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Customer Profile** | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` | • U-shaped response for `age` (<25: $24\%$, 60+: $40\%$ vs 40-49: $8\%$)<br>• High response for students/retirees<br>• Education non-monotonic (`illiterate`: $22\%$, `univ`: $14\%$ vs `basic`: $8\%$)<br>• Housing & personal loans near baseline (~$11\%$) | Liquid capital availability and life-stage financial commitments dictate long-term savings capacity. | • Engineer `age_group` cohorts (`src.transformers.BankFeatureEngineer`)<br>• Standardize continuous `age` via `StandardScaler`<br>• One-Hot Encode all nominal variables including `education` (ADR-002) with `handle_unknown='ignore'` | • Binned age enables linear models to capture non-linear extremes<br>• OHE education avoids false monotonicity and invalid penalty on `unknown`<br>• Low-signal features (`housing`, `loan`) require regularization |
| **Campaign Operational** | `contact`, `month`, `day_of_week`, `campaign` | • Cellular ($14.7\%$) $2.8\times$ higher than telephone ($5.2\%$)<br>• May has $33\%$ volume but $6.4\%$ conversion; Mar/Sep/Oct/Dec have $>40\%$ conversion<br>• Monotonic contact fatigue ($13\%$ at 1 call $\to 3\%$ at $>10$ calls)<br>• Day of week uniform (~$10-12\%$) | Direct personal access via cellular; quarter-end budgeting cycles; telemarketing contact exhaustion. | • Scale `campaign` with `RobustScaler` (heavy right tail up to 56 contacts)<br>• One-Hot Encode `contact`, `month`, `day_of_week`<br>• Stratified split preserves September (ADR-001) | • Robust scaling protects distance and linear models from call count outliers<br>• Evaluation can derive operational decision thresholds (max 3 calls per lead) |
| **Historical Campaign** | `previous`, `pdays`, `poutcome` | • Prior contact jumps conversion ($8.8\%$ at 0 $\to 46\%$ at 2+)<br>• `pdays < 999` achieves $64\%$ vs $9.3\%$ for 999<br>• `poutcome = 'success'` achieves $65.1\%$ ($5.8\times$ lift)<br>• Prior failure still beats cold leads ($14.2\%$) | Warm lead trust equilibrium; proven banking relationship and verified savings capacity. | • Derive binary `previously_contacted` (`pdays != 999`)<br>• Bin recency into `pdays_group` cohorts and drop raw `pdays`<br>• Scale `previous` with `RobustScaler`<br>• One-Hot Encode `poutcome` | • Prevents sentinel 999 from distorting continuous feature space<br>• Subgroup evaluation must verify cold-lead discrimination ($86\%$ of customer base) |
| **Macroeconomic Climate** | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | • Strongest linear correlations with target ($r \approx -0.30$ to $-0.35$)<br>• Conversions surge during low interest rate / negative employment periods<br>• Extreme collinearity ($r > 0.90$ to $0.97$ among `euribor3m`, `emp.var.rate`, `nr.employed`) | Flight to safety during 2008-2010 crisis: risk-averse depositors seek guaranteed fixed bank savings. | • Standardize all 5 economic variables using `StandardScaler` to resolve scale disparity | • Mandatory L2 regularization for linear models to mitigate multicollinearity variance inflation<br>• Tree ensembles partition correlated features without instability<br>• Validates stratified holdout (ADR-001) |

---

# 4. Comprehensive Data Quality, Integrity & Leakage Diagnostics

In accordance with project requirements and the **IT3091 Machine Learning Specification**, an exploratory data analysis must investigate data quality anomalies that impact downstream pipeline design, statistical validity, and predictive reliability.

This section systematically evaluates:
1. **Duplicate Records** (exact duplicate rows and duplicate feature profiles excluding target);
2. **Numerical Range Validation** (domain sanity boundaries, physiological limits, negative values);
3. **Categorical Cardinality & Format Integrity** (unique level counts, spacing, unexpected values);
4. **Unusually Rare Categories** (small-sample instability, split discontinuity risks);
5. **Distribution Skewness & Outliers** (heavy tails in call counts and durations);
6. **Sentinel Value Analysis** (`pdays = 999` domain encoding vs continuous interval assumptions);
7. **Cross-Field Consistency Checks** (logical invariants between `previous`, `pdays`, and `poutcome`);
8. **Leakage-Prone Feature Inspection** (`duration` post-event contamination analysis).

Each diagnostic is accompanied by runnable code, quantitative evidence, and an explicit **Modelling & Preprocessing Implication**.

### 4.1 Duplicate Records Analysis (Exact & Feature-Subset)

Duplicate records can introduce subtle data leakage and evaluation bias if identical customer profiles appear in both training and test partitions. We evaluate:
1. **Exact duplicate rows** across all 21 columns (including target `y`);
2. **Feature duplicates** across the 20 predictor columns (excluding target `y`);
3. **Conflicting target labels** (whether identical feature profiles ever map to contradictory subscription outcomes).

> **Code Comments (Cell 37):**
> - 1. Exact Duplicate Records Check across all 21 columns
> - 2. Duplicate Records Excluding Target Variable 'y'
> - 3. Target Label Consistency Check for Identical Feature Profiles
> - 4. Sample Duplicate Records Inspection

#### Findings & Modelling / Preprocessing Implications: Duplicate Records

- **Empirical Findings**:
  - The dataset contains exactly **12 duplicate rows** (24 rows in total belong to duplicate pairs).
  - Comparing feature subsets excluding target `y` produces identical counts: exactly 12 redundant rows, with **0 conflicting target groups**.
  - All 12 duplicate customer records carry the identical target outcome (`y = 'no'`).
- **Modelling & Preprocessing Implications**:
  1. **Data Leakage Risk Across Splits**: If exact duplicate rows are retained during train/test partitioning, identical customer profiles can fall into both training and holdout sets. This creates subtle data leakage and yields artificially optimistic generalization estimates.
  2. **Loss Function Distortion**: In gradient-based and tree-based algorithms, retaining duplicate rows artificially overweights the repeated feature points in the objective loss function.
  3. **Pipeline Action**: In the preprocessing pipeline, exact duplicates must be pruned before train/test splitting via `df.drop_duplicates(inplace=True)`, reducing dataset size from 41,188 to 41,176 records.

> **Governance Alignment**: Formally ratified under **[ADR-003](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-003)** (Deduplication Policy): exact duplicate rows are pruned prior to train/test splitting (`df.drop_duplicates()`), preserving 41,176 records and eliminating cross-split duplicate leakage.

### 4.2 Numerical Feature Range & Domain Boundary Validation

Each numerical variable is inspected against domain and physical sanity boundaries:
- `age`: Working and retirement demographic bounds $[17, 100]$;
- `duration`: Call duration in seconds ($\ge 0$);
- `campaign`: Contacts within current campaign ($\ge 1$);
- `pdays`: Days elapsed since previous campaign contact ($[0, 999]$, where 999 is sentinel);
- `previous`: Historical contact count ($\ge 0$);
- Macroeconomic indices: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`.

> **Code Comments (Cell 40):**
> - Numerical Feature Range Validation against Expected Domain Boundaries

#### Findings & Modelling / Preprocessing Implications: Numerical Range Validation

- **Empirical Findings**:
  - All 10 numerical features satisfy expected domain boundaries with zero domain violations.
  - No negative values occur in non-negative fields (`age`, `duration`, `campaign`, `pdays`, `previous`).
  - `cons.conf.idx` is entirely negative across all 41,188 observations (range $[-50.8, -26.9]$), which reflects negative consumer sentiment in Portugal during 2008–2010.
  - Dramatic scale divergence exists across features: `nr.employed` ($\sim 5,000$), `duration` ($0$ to $4,918$), `euribor3m` ($0.63$ to $5.05$), and `emp.var.rate` ($-3.4$ to $+1.4$).
- **Modelling & Preprocessing Implications**:
  1. **Disparate Scale & Weight Bias**: Linear models (Logistic Regression), Support Vector Machines, and Neural Networks will assign disproportionate weight to large-magnitude features (`nr.employed`, `duration`) if features are unscaled.
  2. **Pipeline Scaling Strategy**: Standardization via `StandardScaler` is required for macroeconomic indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`) and demographic `age`, while skewed count features (`campaign`, `previous`) benefit from `RobustScaler`.

### 4.3 Categorical Feature Cardinalities & Format Integrity Validation

Categorical variables are evaluated for:
- Cardinality (distinct category count);
- String integrity (casing, leading/trailing whitespace, unexpected non-standard values);
- Operational calendar constraints (represented months and days).

> **Code Comments (Cell 43):**
> - Categorical Feature Cardinality and Integrity Inspection
> - Validate format integrity: clean lowercase strings without extra whitespace

#### Findings & Modelling / Preprocessing Implications: Categorical Cardinalities

- **Empirical Findings**:
  - Cardinality is low-to-moderate across all categorical predictors, ranging from 2 (`contact`) to 12 (`job`).
  - All categorical entries are formatted consistently (lowercase strings without stray whitespace or encoding corruptions).
  - **Operational Calendar Anomalies**:
    - `month` has cardinality **10** (January and February are completely absent from dataset logs).
    - `day_of_week` has cardinality **5** (Monday through Friday only; telemarketing calls were conducted exclusively on business weekdays).
- **Modelling & Preprocessing Implications**:
  1. **One-Hot Encoding Viability**: Low cardinality means standard One-Hot Encoding produces only $pprox 53$ total feature columns, avoiding severe curse-of-dimensionality or sparse memory penalties.
  2. **Out-of-Vocabulary Safety**: Encoders deployed in production must configure `handle_unknown='ignore'` so that unseen categories (e.g. prospective January or weekend contacts) do not crash pipeline scoring.

### 4.4 Unusually Rare Categories & Small-Sample Instability

Categories with extremely low representation ($< 1.0\%$ of the dataset) create statistical instability during cross-validation and risk severe coefficient variance in linear classifiers.

> **Code Comments (Cell 46):**
> - Scan for unusually rare category levels (< 1.0% prevalence)

#### Findings & Modelling / Preprocessing Implications: Rare Categories

- **Empirical Findings**:
  - `default = 'yes'`: Contains only **3 observations** out of 41,188 ($0.007\%$), with 0 subscribers ($0.00\%$ response rate).
  - `education = 'illiterate'`: Contains only **18 observations** ($0.044\%$), with 4 subscribers ($22.22\%$ response rate).
  - `marital = 'unknown'`: Contains **80 observations** ($0.194\%$), with 12 subscribers ($15.00\%$ response rate).
  - `month = 'dec'`: Contains **182 observations** ($0.442\%$), with 89 subscribers ($48.90\%$ response rate).
  - `job = 'unknown'`: Contains **330 observations** ($0.801\%$), with 37 subscribers ($11.21\%$ response rate).
- **Modelling & Preprocessing Implications**:
  1. **Validation Split Discontinuity**: For `default = 'yes'` ($N=3$), any 5-fold cross-validation or holdout split will leave multiple folds with zero occurrences, making fold metrics unstable.
  2. **Quasi-Complete Separation & Weight Explosion**: In unregularized logistic regression, a binary column with 0 positive responses (like `default = 'yes'`) pushes its weight toward $-\infty$. Strong L2 regularization (`C=1.0` or smaller) is essential.
  3. **Information Role of `default`**: Because `default = 'yes'` has only 3 cases while `default = 'no'` has 32,588 and `default = 'unknown'` has 8,597 ($20.87\%$), the feature primarily operates as an indicator of **missing credit history**, not credit risk.

### 4.5 Numerical Outlier Detection & Skewness Analysis

Numerical variables are evaluated for heavy-tailed skewness and extreme values using:
- **IQR Rule**: Values outside $[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$;
- **Z-Score Rule**: Values exceeding $|Z| > 3$ standard deviations;
- **Distribution Skewness**: Fisher-Pearson skewness coefficient.

> **Code Comments (Cell 49):**
> - Outlier and Skewness Inspection Across Numerical Predictors

> **Code Comments (Cell 50):**
> - Visualizing Distributions and Outliers for Highly Skewed Features

#### Findings & Modelling / Preprocessing Implications: Outliers & Skewness

- **Empirical Findings**:
  - `campaign`: Substantial right-skew (skewness $= 4.76$), with values reaching $56$ contacts while the 99th percentile is $14$ contacts.
  - `duration`: High right-skew (skewness $= 3.26$), with maximum duration of $4,918$ seconds ($pprox 82$ minutes) against a median of $180$ seconds.
  - `age`: Moderate right-skew (skewness $= 0.78$), extending up to $98$ years, with $469$ records flagged by the IQR rule ($> 69$ years).
- **Modelling & Preprocessing Implications**:
  1. **Harm of Naive Outlier Truncation**: Truncating elderly clients (`age > 60`, $N = 1,020$) based on IQR bounds would severely degrade model capability, because seniors and retirees exhibit the highest conversion rates in the dataset ($> 45\%$).
  2. **Robust Scaler Selection**: For count variables like `campaign` and `previous`, `RobustScaler` (centering by median, scaling by IQR) prevents extreme contact counts from collapsing the variance of standard customer profiles.
  3. **Tree Model Invariance**: Decision trees and ensemble trees (Random Forest, LightGBM) split on monotonic order rather than metric Euclidean distance, rendering them immune to outlier distortions in `campaign` and `age`.

### 4.6 Sentinel Value Analysis: `pdays = 999`

In the UCI dataset documentation, `pdays` denotes the number of days that passed since the client was last contacted from a previous campaign. The value `999` is an explicit sentinel code indicating that the client **was not previously contacted**.

> **Code Comments (Cell 53):**
> - Sentinel Value Quantification: pdays == 999 vs Genuine Day Intervals

> **Code Comments (Cell 54):**
> - Visualizing pdays Distribution and Conversion Rate Contrast
> - Distribution for contacted clients
> - Conversion rate contrast

#### Findings & Modelling / Preprocessing Implications: Sentinel `pdays = 999`

- **Empirical Findings**:
  - **39,673 records ($96.32\%$)** carry the sentinel value `999`.
  - Only **1,515 records ($3.68\%$)** carry genuine day counts, spanning $[0, 27]$ days (mean $6.01$ days, median $6.0$ days).
  - Massive conversion gap: Clients with `pdays = 999` have a conversion rate of **$9.26\%$**, while clients with `pdays < 999` convert at **$63.83\%$** (nearly a 7-fold increase).
- **Modelling & Preprocessing Implications**:
  1. **Severe Metric Distortion**: Treating `999` as a continuous numeric interval causes linear models, KNN, and neural networks to treat uncontacted clients as having been contacted 999 days ago ($pprox 2.7$ years), forcing an arbitrary linear penalty.
  2. **Variance Compression under Standardization**: Standard scaling yields mean $962.5$ and std $186.9$, compressing genuine differences in $[0, 27]$ into a negligible interval $[-5.15, -5.00]$ while placing sentinel clients at $+0.20$.
  3. **Pipeline Transformation Strategy**:
     - Derive a binary indicator: `previously_contacted = (pdays != 999).astype(int)`.
     - Discretize valid contact days into categorical bins (`0_to_6_days`, `7_to_14_days`, `15_plus_days`, `not_contacted`).
     - Drop the raw numeric `pdays` column to eliminate scale distortion. This architecture is encapsulated in `src.transformers.BankFeatureEngineer`.

### 4.7 Cross-Field Consistency Checks: `previous`, `pdays`, and `poutcome`

The dataset records customer campaign history across three distinct attributes:
- `previous`: Number of contacts prior to this campaign ($0, 1, 2, \dots$);
- `pdays`: Days elapsed since last previous contact ($[0, 27]$ or $999$);
- `poutcome`: Outcome of previous marketing campaign (`nonexistent`, `failure`, `success`).

We audit the cross-field logical invariants between these fields to uncover data logging conventions and potential integrity issues.

> **Code Comments (Cell 57):**
> - Cross-Tabulation Consistency Diagnostics
> - Investigate anomalous cohort: previous > 0 BUT pdays == 999
> - Summary Table of Consistency Rules

#### Findings & Modelling / Preprocessing Implications: Cross-Field Consistency

- **Empirical Findings**:
  - **Perfect Invariant 1**: Every record with `previous == 0` ($35,563$ records) has `poutcome == 'nonexistent'` and `pdays == 999` with zero exceptions.
  - **Perfect Invariant 2**: Every record with `poutcome == 'success'` ($1,373$ records) has a recorded `pdays < 999` ($[0, 27]$ days).
  - **Critical Domain Discrepancy**: Out of $5,625$ clients who were previously contacted (`previous > 0`), **4,110 clients ($73.07\%$)** have `pdays == 999`!
    - Dissection reveals that **$100.0\%$ of these 4,110 clients** have `poutcome == 'failure'`.
    - Domain Explanation: The telemarketing CRM only logged exact day intervals (`pdays`) if the previous outreach was recent or resulted in subscription; if the prior campaign ended in failure and occurred in an earlier campaign season, the elapsed days counter was unrecorded, defaulting to `999`.
- **Modelling & Preprocessing Implications**:
  1. **Feature Engineering Trap**: If a pipeline engineers a historical contact flag solely as `is_contacted = (pdays != 999)`, it will **misclassify 4,110 previously contacted customers as brand-new leads**, ignoring historical contact attempts.
  2. **True Contact Boundary**: The true historical interaction boundary is `previous > 0` (or `poutcome != 'nonexistent'`).
  3. **Information Preservation**: Because `pdays == 999` merges genuine new leads ($N=35,563$) with historically failed contacts ($N=4,110$), the model MUST retain `previous` and `poutcome` alongside any `pdays` transformations to allow tree and linear classifiers to disambiguate the two groups.

> **Governance Alignment**: Formally ratified under **[ADR-003](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-003)** (Cross-Field Invariant Policy): historical campaign features (`previous` count and `poutcome`) are preserved alongside binned `pdays` to avoid misclassifying the 4,110 legacy CRM failures.

### 4.8 Leakage-Prone Feature Inspection: Call `duration`

The UCI Bank Marketing dataset notes regarding `duration`:
> *"This attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model."*

We inspect the quantitative relationship between `duration` and `y` to demonstrate why retaining it constitutes post-hoc target leakage.

> **Code Comments (Cell 60):**
> - Quantitative Inspection of Call Duration and Data Leakage
> - Zero-duration calls check
> - Conversion rate progression across call duration brackets

> **Code Comments (Cell 61):**
> - Visualizing Call Duration Distribution and Conversion Progression
> - Density by outcome
> - Conversion rate by duration bracket

#### Findings & Modelling / Preprocessing Implications: Call `duration` Leakage

- **Empirical Findings**:
  - `duration` is by far the strongest individual predictor in the dataset, with a linear correlation of $+0.405$ with $y$.
  - Median call duration for subscribers is **$449$ seconds ($pprox 7.5$ minutes)** versus **$163.5$ seconds ($pprox 2.7$ minutes)** for non-subscribers.
  - Conversion rate escalates dramatically with duration: calls $\le 1$ minute yield a **$0.02\%$ conversion rate** ($1$ subscriber out of $4,286$ calls), whereas calls $> 10$ minutes yield a **$48.61\%$ conversion rate**.
  - All $4$ calls with `duration == 0` resulted in `no`.
- **Modelling & Preprocessing Implications**:
  1. **Severe Post-Event Target Leakage**: In real outbound telemarketing operations, machine learning models are deployed to **score and rank leads before placing calls**. The call duration is unknown at the moment of lead selection.
  2. **Mechanistic Consequence Rather Than Predictor**: High duration is an operational symptom of customer interest (reading contracts, answering compliance questions, setting up accounts), not an independent driver.
  3. **Catastrophic Production Failure**: A model trained with `duration` will rely on it almost exclusively, achieving artificially inflated validation metrics ($> 0.93$ ROC-AUC). When deployed for prospect selection where duration is unavailable, the model will fail completely.
  4. **Strict Architectural Exclusion**: In compliance with Constitution Principle III (*No Data Leakage*) and official UCI directives, `duration` MUST be completely excluded from the predictive feature pipeline (`remainder='drop'` in `ColumnTransformer`).

> **Governance Alignment**: Formally ratified under **[ADR-003](file:///Users/moni/Desktop/ML%20proj/docs/decision_log.md#adr-003)** (Data Leakage Quarantine Policy) and Constitution Principle III (*No Data Leakage*): call `duration` is excluded from all operational feature matrices.

### 4.9 Synthesis of Data Quality Findings & Preprocessing Action Matrix

| Data Quality Dimension | Empirical Finding | Modelling & Evaluation Risk | Preprocessing Action (Phase 2 Pipeline) |
| :--- | :--- | :--- | :--- |
| **Exact Duplicates** | 12 duplicate rows ($0.029\%$), 0 conflicting targets | Data leakage across train/test splits; objective function over-weighting | Drop exact duplicates (`df.drop_duplicates()`) before splitting |
| **Numerical Ranges** | All 10 features satisfy physical/domain bounds | Extreme scale divergence (`nr.employed` $\sim 5000$ vs `euribor3m` $\sim 1-5$) dominates linear models | Standardize via `StandardScaler`; scale skewed features with `RobustScaler` |
| **Categorical Cardinality** | Low-to-moderate cardinalities ($2$ to $12$); Jan/Feb and weekends absent | Combinatorial growth if unmanaged; unobserved categories crash inference | One-Hot Encoding with `handle_unknown='ignore'`; drop collinear reference if linear |
| **Rare Categories** | `default='yes'` ($N=3$), `illiterate` ($N=18$), `dec` ($N=182$) | Zero-count splits in cross-validation; infinite logit weights | Apply L2 regularization; use `handle_unknown='ignore'`; treat `default` as unknown-status indicator |
| **Outliers & Heavy Tails** | Extreme right skew in `campaign` (max $56$) and `duration` (max $4,918$s) | Standard deviation distortion; linear model sensitivity | Apply `RobustScaler` to `campaign` and `previous`; retain elder age groups (conversion $> 45\%$) |
| **Sentinel Values** | `pdays = 999` in $96.32\%$ of records; $3.68\%$ valid days ($[0, 27]$) | 999 treated as continuous distance distorts regression planes and clustering | Feature engineer `previously_contacted` binary flag and `pdays_group` categorical bins; drop raw `pdays` |
| **Cross-Field Consistency** | $4,110$ records have `previous > 0` but `pdays = 999` (all `poutcome='failure'`) | Deriving historical contact only from `pdays != 999` misclassifies $4,110$ customers | Retain `previous` and `poutcome` alongside `pdays` transformations to preserve interaction history |
| **Leakage-Prone Features** | `duration` correlates $+0.405$ with $y$; unknown prior to placing calls | Severe target leakage; produces model that cannot score leads at inference time | Drop `duration` entirely from operational feature matrix (`remainder='drop'`) |

---

# 5. Core Evidence: Structured EDA Insight Log

In accordance with project deliverables, this structured evidence log synthesizes all primary exploratory findings into actionable machine learning engineering directives. Each finding directly couples an empirical dataset observation to its downstream modeling or preprocessing decision.

### Master EDA Insight Log

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

> **Canonical Documentation Reference**: This log is maintained as a formal project deliverable in [`docs/eda_insight_log.md`](../docs/eda_insight_log.md) and governs the preprocessing pipeline in [`notebooks/02_Data_Preprocessing_Pipeline.ipynb`](02_Data_Preprocessing_Pipeline.ipynb) and architectural decisions in [`docs/decision_log.md`](../docs/decision_log.md).

