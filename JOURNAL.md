# 🩺 Medical Cost Predictor & ML Engineering Journal

An end-to-end learning log and architectural record for building, optimizing, and deploying a production-ready healthcare cost prediction system with Scikit-Learn and Streamlit.

---

## 📋 Table of Contents
1. [Executive Summary](#executive-summary)
2. [Milestone 0: Project Setup & Environment Isolation](#milestone-0-project-setup--environment-isolation)
3. [Milestone 1: Exploratory Data Analysis (EDA) & Problem Formulation](#milestone-1-exploratory-data-analysis-eda--problem-formulation)
4. [Milestone 2: Baseline Model (Linear Regression) & Data Splitting](#milestone-2-baseline-model-linear-regression--data-splitting)
5. [Milestone 3: Non-Linear Models (Random Forest Regressor)](#milestone-3-non-linear-models-random-forest-regressor)
6. [Milestone 4: Pipeline Encapsulation & Model Serialization](#milestone-4-pipeline-encapsulation--model-serialization)
7. [Milestone 5: Interactive Web Application (Streamlit)](#milestone-5-interactive-web-application-streamlit)
8. [Milestone 6: Model Evolution, Hyperparameter Tuning & Case Grounding](#milestone-6-model-evolution-hyperparameter-tuning--case-grounding)
9. [Summary of Performance Benchmarks](#summary-of-performance-benchmarks)

---

## Executive Summary
* **Problem**: Predict annual individual medical charges based on demographic and clinical indicators.
* **Dataset**: 1,338 US patient records (6 features: `age`, `sex`, `bmi`, `children`, `smoker`, `region`).
* **Initial Baseline (Linear Regression)**: $\text{MAE} = \$4,181.19$ | $R^2 = 78.36\%$
* **Final Champion (Tuned Gradient Boosting)**: $\text{MAE} = \mathbf{\$2,392.89}$ | $R^2 = \mathbf{88.04\%}$
* **Key Achievements**:
  * Reduced average prediction error by **\$1,788.30 per patient (-42.8%)**.
  * Increased explained variance by **+9.68%**.
  * Encapsulated full preprocessing inside an atomic Scikit-Learn `Pipeline` to prevent training/serving skew.
  * Deployed an interactive Streamlit UI with **Historical Case Grounding (Nearest Neighbors)** and counterfactual **What-If lifestyle simulations**.

---

## Milestone 0: Project Setup & Environment Isolation

### 1. Python Virtual Environment (`venv`)
* **Purpose**: Creates an isolated sandbox containing its own Python interpreter and package dependencies.
* **Why it matters**:
  * Prevents version collisions between different projects.
  * Ensures reproducibility and portability across operating systems.
  * Keeps the host system's global Python installation clean.

```bash
# Initialize virtual environment
python -m venv costpredictor

# Activate virtual environment
source costpredictor/bin/activate
```

### 2. Dependency Management (`requirements.txt`)
* A plain-text manifest pinning project dependencies.

```bash
pip install -r requirements.txt
```

```text
pandas
numpy
scikit-learn
streamlit
joblib
```

### 3. Dataset Acquisition
* Downloaded the standard US Medical Cost dataset (1,338 records):

```bash
curl https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv -o data/insurance.csv
```

---

## Milestone 1: Exploratory Data Analysis (EDA) & Problem Formulation

EDA is the process of critically examining the dataset before writing modeling code to understand schema, distributions, outliers, and data integrity.

### 1. Target ($y$) vs. Features ($X$)
* **Target ($y$)**: The ground-truth variable we aim to predict $\to$ `charges` (continuous dollar amount).
* **Model Prediction ($\hat{y}$)**: The estimated value output by the machine learning algorithm.
* **Features ($X$)**: The input signals used for estimation $\to$ `age`, `sex`, `bmi`, `children`, `smoker`, `region`.

### 2. Problem Formulation: Regression vs. Classification
* **Regression**: Predicting a continuous numeric quantity (e.g., hospital bill in dollars).
* **Classification**: Predicting discrete categorical labels (e.g., Cat vs. Dog, Malignant vs. Benign).
* *Conclusion*: This project is a **Regression** problem.

### 3. Feature Categorization
* **Numerical Features** (Have mathematical meaning):
  * **Discrete**: Formed by counting whole units $\to$ `children` ($0, 1, 2, \dots$).
  * **Continuous**: Formed by measurement with infinite decimal precision $\to$ `age`, `bmi` (e.g., $27.9$, $33.77$).
* **Categorical Features** (Discrete group labels without inherent mathematical ordering):
  * `sex` (`male`, `female`)
  * `smoker` (`yes`, `no`)
  * `region` (`southwest`, `southeast`, `northwest`, `northeast`)

### 4. Data Hygiene: Garbage In, Garbage Out (GIGO)
* If training data contains impossible anomalies (e.g., negative ages or BMI of 900), the algorithm treats them as valid patterns.
* EDA confirmed:
  * Total missing values: **0** across all columns.
  * Charges range: \$1,121.87 to \$63,770.43 (median: \$9,382.03, mean: \$13,270.42).
  * **The Smoking Disparity**:
    * Average Non-Smoker Charge: **\$8,434.27**
    * Average Smoker Charge: **\$32,050.23**
    * **Cost Gap: +\$23,615.96** (smoking status is the primary cost driver).

---

## Milestone 2: Baseline Model (Linear Regression) & Data Splitting

Always build a simple baseline first to establish a performance benchmark.

### 1. The Golden Rule: Train / Test Split
* **The Exam Analogy**: Giving a model the exact test questions to study proves *memorization* (**Overfitting**), not *generalization*.
* **Split Configuration**:
  * **Training Set (80%)**: Used by the algorithm to compute weights and bias.
  * **Test Set (20%)**: Unseen data used exclusively for objective evaluation.
  * **`random_state=42`**: Ensures deterministic, reproducible data shuffling.

### 2. Encoding Categorical Variables
* Mathematical equations cannot operate on raw text strings (`"yes"`, `"female"`).
* **Why not label 1, 2, 3, 4?** Assigning numbers introduces false mathematical rankings (e.g., implying `northeast (4) = 2 × southeast (2)`).
* **One-Hot Encoding**: Converts categories into binary $0/1$ indicator columns.
  * `drop_first=True`: Drops redundant reference columns to prevent multicollinearity (the **dummy variable trap**) and reduce memory.

### 3. Evaluation Metrics
* **MAE (Mean Absolute Error)**: Average dollar magnitude of errors:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
  * *Scale*: \$0.0 is perfect.
* **$R^2$ (Coefficient of Determination)**: Proportion of variance explained by the model:
  * $1.0$ (100%): Perfect predictions.
  * $0.0$: Performs no better than predicting the dataset mean for everyone.
  * $< 0.0$: Worse than guessing the mean.

### 4. Baseline Results
* **MAE**: **\$4,181.19** (On average, predictions are off by $\pm \$4,181$)
* **$R^2$**: **0.7836 (78.36%)**

---

## Milestone 3: Non-Linear Models (Random Forest Regressor)

### 1. Why Linear Models Plateau: Feature Interactions
* Linear Regression assumes linear independence:
  $$\hat{y} = w_1(\text{BMI}) + w_2(\text{Smoker}) + \dots + b$$
* In healthcare, risk factors **interact non-linearly**:
  * High BMI alone increases costs moderately.
  * Smoking alone increases costs moderately.
  * **High BMI ($\ge 30$) + Smoking** creates an explosive multiplicative risk surge. Linear models cannot capture this without manual mathematical features.

### 2. Random Forest Regressor (Ensemble Bagging)
* **Decision Tree**: Splits data using conditional IF/ELSE questions on features. Single trees often overfit.
* **Random Forest**: Builds 100 decision trees (`n_estimators=100`) on bootstrapped random subsets of data and features.
* **Regression Aggregation**: Takes the **mathematical mean** of all 100 individual tree predictions (canceling out individual tree errors).

### 3. Random Forest Results
* **MAE**: **\$2,544.64** (Reduced error by **\$1,636.55** vs. Linear Regression)
* **$R^2$**: **0.8653 (86.53%)** (+8.17% variance explained)

---

## Milestone 4: Pipeline Encapsulation & Model Serialization

### 1. The Production Trap (Why `pd.get_dummies` Fails in Production)
* `pd.get_dummies()` works in offline scripts because it inspects all 1,338 rows at once.
* In a production app, a single new patient enters data:
  ```python
  {"age": 30, "sex": "female", "bmi": 26.5, "children": 0, "smoker": "no", "region": "northwest"}
  ```
* Applying `pd.get_dummies()` to this single row produces only 2–3 columns instead of the 8 columns expected by the trained model.
* The script crashes with: `ValueError: Feature shape mismatch`.

### 2. The Solution: `ColumnTransformer` + `Pipeline`
* **`ColumnTransformer`**: Directs specific transformations to designated columns:
  * Categorical columns (`sex`, `smoker`, `region`) $\to$ `OneHotEncoder(handle_unknown='ignore', drop='first')`.
  * Numerical columns (`age`, `bmi`, `children`) $\to$ `'passthrough'`.
* **`Pipeline`**: Chains the preprocessor and estimator together as an atomic unit:
  ```
  [Raw Input Data] ──> [ColumnTransformer] ──> [RandomForestRegressor] ──> [Predicted Cost ($)]
  ```
* The pipeline automatically handles transformation during inference on raw dictionaries/DataFrames.

### 3. Model Serialization (`joblib`)
* Converts the live in-memory Python pipeline into a persisted binary file on disk (`models/model.joblib`).
* Enables the web application to load the trained model in milliseconds without retraining.

---

## Milestone 5: Interactive Web Application (Streamlit)

### 1. Streamlit Architecture & Execution Model
* Streamlit converts Python scripts into interactive web applications.
* **Execution Flow**: Scripts re-run top-to-bottom on every widget interaction.
* **In-Memory Caching (`@st.cache_resource`)**:
  * Prevents reading the 10 MB serialized model from disk on every slider movement.
  * Caches the loaded pipeline in RAM once across user sessions for sub-second inference.

### 2. Counterfactual "What-If" Health Risk Simulation
* Beyond point predictions, the app provides clinical decision support:
  * **Smoking Cessation**: Generates a parallel prediction setting `smoker='no'` to quantify annual savings (\$15k–\$25k/year).
  * **BMI Optimization**: Generates a parallel prediction setting `bmi=24.0` (normal WHO weight range) to quantify savings from weight management.

---

## Milestone 6: Model Evolution, Hyperparameter Tuning & Case Grounding

### 1. Model 2: Gradient Boosting (`GradientBoostingRegressor`)
* **Bagging vs. Boosting**:
  * Random Forest trains 100 trees **in parallel independently** and averages them.
  * Gradient Boosting trains trees **sequentially in a chain**:
    * Tree #1 makes predictions and calculates **residuals** ($\text{Actual} - \text{Predicted}$).
    * Tree #2 fits specifically on the residuals of Tree #1.
    * Subsequence trees progressively correct remaining errors along the gradient.
* **Result**: MAE dropped to **\$2,446.20** | $R^2$ reached **87.92%**.

### 2. Model 3: Feature Engineering (`is_obese_smoker`)
* Engineered explicit clinical interaction:
  $$\text{is\_obese\_smoker} = (\text{BMI} \ge 30.0) \land (\text{Smoker} == \text{'yes'})$$
* **Result**: MAE \$2,486.91 | $R^2$ 87.52%.
* **Insight**: Gradient Boosting already isolated this interaction natively through sequential splits. Adding an explicit collinear binary column produced slight feature redundancy.

### 3. Model 4: Tuned Champion via `GridSearchCV`
* Automated grid search over 54 parameter combinations using **5-Fold Cross Validation**:
  * Tested: `n_estimators` ($[100, 150, 200]$), `learning_rate` ($[0.03, 0.05, 0.1]$), `max_depth` ($[2, 3, 4]$), `min_samples_split` ($[2, 5]$).
* **Winning Configuration**:
  ```python
  {
      'model__learning_rate': 0.1,
      'model__max_depth': 2,
      'model__min_samples_split': 5,
      'model__n_estimators': 150
  }
  ```
* **Key Insight**: Shallower trees (`max_depth: 2` vs. default `3`) won. Shallow trees prevent individual trees from memorizing training noise, maximizing test set generalization.
* **Final Champion Score**: **MAE \$2,392.89** | **$R^2$ 88.04%**.

### 4. Theoretical Boundaries: Why Can't We Reach 99% $R^2$? (Irreducible Error)
* In statistical learning:
  $$y = f(X) + \epsilon$$
* **The "Identical Twin" Paradox**: Two patients can have identical age, sex, BMI, children, and smoker status. Yet one goes for a routine \$1,200 checkup while the other suffers an emergency accident costing \$35,000.
* Without unobserved latent features (procedure codes, diagnosis codes, hospital pricing tiers), that variance cannot mathematically be predicted from demographic data alone.
* **Industry Standard**: An $R^2$ of ~88% is considered an exceptional ceiling for demographic-only healthcare economics. Higher scores without diagnosis features indicate data leakage or severe overfitting.

### 5. Case-Based Reasoning: Grounding with Nearest Neighbors
* **Explainability Problem**: ML predictions can feel like a "black box" to users and clinicians.
* **Solution**: Deployed Scikit-Learn's `NearestNeighbors` over weighted, normalized clinical feature space:
  * Retrieves the #1 closest historical patient record from the 1,338-cohort in $< 1\text{ ms}$.
  * Displays the matched patient's **Actual Historical Bill** side-by-side with the model's estimate, providing concrete evidentiary grounding.

---

## Summary of Performance Benchmarks

All models were evaluated on the exact same 20% unseen test set (268 patients) using `random_state=42`:

| Milestone / Iteration | Architecture | Test MAE ($) | Test $R^2$ Score | Error Reduction vs. Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Milestone 2 (Baseline)** | Linear Regression | \$4,181.19 | 78.36% | Baseline |
| **Milestone 3** | Random Forest (100 Trees) | \$2,544.64 | 86.53% | -\$1,636.55 (-39.1%) |
| **Milestone 6 (Model 2)** | Gradient Boosting (Default) | \$2,446.20 | 87.92% | -\$1,734.99 (-41.5%) |
| **Milestone 6 (Model 3)** | GB + Feature Engineering | \$2,486.91 | 87.52% | -\$1,694.28 (-40.5%) |
| **Milestone 6 (Champion)** | **Tuned Gradient Boosting (GridSearchCV)** | **\$2,392.89** | **88.04%** | **-\$1,788.30 (-42.8%)** |

---

*Authored during the Machine Learning Engineering pairing sessions.*