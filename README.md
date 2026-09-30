# AI-Powered Predictive Maintenance & Failure Risk Analysis

## Project Overview

This project demonstrates an engineering-focused machine learning workflow for predictive maintenance and machine failure risk analysis.

The project combines mechanical engineering knowledge with Python, data analysis, feature engineering and machine learning to identify operating conditions associated with higher machine-failure risk.

The analysis uses the UCI AI4I 2020 Predictive Maintenance Dataset containing 10,000 observations.

## Project Objectives

- Analyze machine operating data using Exploratory Data Analysis (EDA).
- Identify relationships between operating conditions and machine failure.
- Create meaningful engineering-derived features.
- Build machine-failure prediction models.
- Handle imbalanced failure data.
- Compare a baseline model with a nonlinear machine-learning model.
- Identify important predictive features.
- Generate machine failure-risk categories.
- Translate machine-learning results into a practical predictive-maintenance use case.

## Engineering Problem

Unexpected equipment failure can cause production interruption, maintenance costs and equipment downtime.

Predictive maintenance uses equipment operating data to estimate failure risk before a failure occurs. This can help maintenance teams prioritize inspections and maintenance activities.

This project demonstrates a machine-learning prototype for that purpose.

## Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Source:** UCI Machine Learning Repository

**Observations:** 10,000

**Original columns:** 14

**Machine failures:** 339

**Failure rate:** 3.39%

**Missing values:** 0

The dataset is synthetic and was designed to reflect predictive-maintenance data encountered in industry.

## Data Quality Analysis

The following checks were performed:

- Dataset dimensions
- Column names and data types
- Missing-value analysis
- Duplicate-row analysis
- Duplicate identifier analysis
- Machine-failure distribution
- Failure rate by machine type
- Engineering-variable distributions
- Failure-mode consistency

### Data Quality Results

| Check | Result |
|---|---:|
| Total observations | 10,000 |
| Original columns | 14 |
| Missing cells | 0 |
| Duplicate rows | 0 |
| Duplicate UDI values | 0 |
| Machine failures | 339 |
| Failure rate | 3.39% |

## Engineering Feature Engineering

Three engineering-derived features were created.

### 1. Temperature Difference

Temperature Difference represents the difference between process temperature and air temperature.

```text
Temperature Difference = Process Temperature - Air Temperature
```

### 2. Angular Speed

Rotational speed was converted from RPM to radians per second.

```text
Angular Speed = 2 × π × RPM / 60
```

### 3. Mechanical Power

Mechanical power was calculated using torque and angular speed.

```text
Mechanical Power (kW) = Torque (N·m) × Angular Speed (rad/s) / 1000
```

Mechanical power provides an engineering-relevant representation of the machine's mechanical operating demand.

## Exploratory Data Analysis

Several important patterns were identified.

### Machine Failure

Machine failure is a rare event in the dataset.

```text
339 / 10,000 = 3.39%
```

Therefore, class imbalance was an important consideration during model development.

### Power Regimes

Failure rates by mechanical-power quartile were:

| Power Regime | Observed Failure Rate |
|---|---:|
| Low | 1.96% |
| Medium-Low | 0.88% |
| Medium-High | 1.80% |
| High | 8.92% |

The highest-power regime showed a noticeably higher observed failure rate.

### Power and Tool Wear

The combined High Power + High Tool Wear region showed an observed failure rate of **17.64%**.

These results represent associations observed in the dataset and should not be interpreted as proof of physical causation.

## Machine Learning Approach

The following features were used for predictive modeling:

- Type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Temperature Difference
- Angular Speed
- Mechanical Power

### Features Excluded

**UDI and Product ID**

These are identifiers rather than physical operating variables.

**Machine failure**

This is the prediction target.

**TWF, HDF, PWF, OSF and RNF**

These failure-mode indicators were excluded because they are closely related to the machine-failure target and may represent information available only during or after a failure event.

This helps reduce the risk of target leakage.

## Train/Test Strategy

The dataset was divided into:

- 80% training data
- 20% testing data

A stratified split was used because machine failure is a rare class.

```text
Training samples: 8,000
Testing samples: 2,000
Random state: 42
```

## Model 1 — Logistic Regression

Logistic Regression was used as the baseline model.

Preprocessing included:

- StandardScaler for numerical features
- OneHotEncoder for the categorical Type variable
- Balanced class weights

### Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.934 |
| PR-AUC | 0.466 |
| Failure Precision | 0.177 |
| Failure Recall | 0.868 |
| Failure F1 | 0.294 |

### Confusion Matrix

```text
[[1658, 274],
 [9, 59]]
```

The baseline model detected many actual failures but produced a relatively high number of false alarms.

## Model 2 — Random Forest

Random Forest was used as the nonlinear machine-learning model.

The model configuration was:

```text
Number of trees: 300
Class weight: balanced
Minimum samples per leaf: 2
Random state: 42
```

### Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.963 |
| PR-AUC | 0.856 |
| Failure Precision | 0.909 |
| Failure Recall | 0.735 |
| Failure F1 | 0.813 |
| Accuracy | 0.989 |

### Confusion Matrix

```text
[[1927, 5],
 [18, 50]]
```

## Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.934 | 0.466 |
| Random Forest | 0.963 | 0.856 |

Random Forest produced stronger predictive performance on the held-out test set.

PR-AUC is particularly useful in this project because machine failure is a rare minority class.

## Feature Importance

The main Random Forest feature-importance results were:

| Feature | Importance |
|---|---:|
| Tool wear | 0.1975 |
| Rotational speed | 0.1710 |
| Torque | 0.1687 |
| Mechanical Power | 0.1648 |
| Angular Speed | 0.1375 |
| Temperature Difference | 0.0708 |
| Air temperature | 0.0436 |
| Process temperature | 0.0323 |

Tool wear, rotational speed, torque and mechanical power were among the strongest predictive features in the model.

**Important:** Feature importance indicates predictive contribution within the model. It does not prove that a feature causes machine failure.

## Predictive Maintenance Risk Scoring

The Random Forest model generated a predicted probability of machine failure for each test observation.

These probabilities were converted into three illustrative risk categories:

- Low Risk
- Medium Risk
- High Risk

### Test-set Risk Distribution

| Risk Level | Observations | Observed Failure Rate |
|---|---:|---:|
| Low Risk | 1,905 | 0.52% |
| Medium Risk | 40 | 20.00% |
| High Risk | 55 | 90.91% |

The risk groups demonstrate how machine-learning probabilities can be converted into a practical maintenance-prioritization workflow.

The thresholds used in this portfolio project are illustrative and would require validation against actual maintenance costs and operational requirements before production use.

## Practical Predictive Maintenance Application

A real predictive-maintenance system could use this type of model to:

1. Collect machine operating data.
2. Calculate failure probability.
3. Identify higher-risk equipment.
4. Prioritize inspection.
5. Support maintenance planning.
6. Combine ML predictions with sensor trends and maintenance history.
7. Help maintenance teams make data-driven decisions.

The model should be considered a decision-support system and should not replace engineering inspection or safety procedures.

## Target Leakage Consideration

Target leakage occurs when information unavailable at prediction time is provided to the model.

The failure-mode indicators were excluded because they are closely connected to the failure target and may represent information generated during or after a failure event.

Identifiers were also excluded because they do not represent physical machine operating conditions.

This makes the predictive feature set more appropriate for an early-warning scenario.

## Key Engineering Insights

The project demonstrated that:

- Machine failure is a rare event in the dataset.
- Tool wear is an important predictive feature.
- Rotational speed and torque are important operating variables.
- Mechanical power provides an engineering-relevant derived feature.
- Higher-power operating conditions showed higher observed failure rates.
- High power combined with high tool wear showed elevated observed failure rates.
- Random Forest captured nonlinear relationships between machine operating variables and failure risk.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Logistic Regression
- Random Forest
- Exploratory Data Analysis
- Feature Engineering
- Imbalanced Classification
- Predictive Maintenance
- Risk Scoring
- Mechanical Engineering Analysis

## Project Structure

```text
AI-Powered-Predictive-Maintenance/
│
├── README.md
│
└── 01_predictive_maintenance_analysis.ipynb
```

## Notebook

The complete analysis and machine-learning workflow is available in:

`01_predictive_maintenance_analysis.ipynb`

The notebook contains:

- Data loading
- Data-quality checks
- Exploratory Data Analysis
- Engineering feature creation
- Operating-regime analysis
- Feature preparation
- Logistic Regression
- Random Forest
- Model evaluation
- Feature importance
- Risk scoring
- Engineering conclusions

## Kaggle Project

The complete interactive analysis is available on my public Kaggle profile.

**Kaggle Notebook:**

https://www.kaggle.com/code/sayoojkp741/ai-powered-predictive-maintenance-data-eda

## Project Limitations

This is a portfolio machine-learning prototype and not a production predictive-maintenance system.

Important limitations include:

- The dataset is synthetic.
- Validation used a single stratified train/test split.
- Real industrial deployment would require historical plant data.
- Time-based validation may be required for real equipment data.
- Risk thresholds would need business validation.
- Feature importance does not establish causality.
- Production deployment would require monitoring and model-drift detection.

## Future Improvements

Possible improvements include:

- Cross-validation
- Time-based validation
- Probability calibration
- Threshold optimization based on maintenance costs
- SHAP-based model explainability
- Real sensor/IoT data integration
- CMMS/SAP maintenance-data integration
- Model monitoring
- Data-drift detection
- Real-world plant validation

## Engineering + AI Perspective

This project demonstrates the combination of mechanical engineering and artificial intelligence.

### Mechanical Engineering

- Machine operating conditions
- Torque
- Rotational speed
- Tool wear
- Mechanical power
- Maintenance reasoning

### Artificial Intelligence / Machine Learning

- Python
- Data analysis
- Feature engineering
- Classification
- Imbalanced learning
- Model evaluation
- Risk prediction

This combination is relevant to manufacturing, industrial AI, predictive maintenance and engineering analytics applications.

## Dataset Attribution

**AI4I 2020 Predictive Maintenance Dataset**

Matzka, S. (2020).

UCI Machine Learning Repository.

DOI: 10.24432/C5HS5C

## Project Results

### Random Forest

```text
ROC-AUC: 0.963
PR-AUC: 0.856
Failure Precision: 0.909
Failure Recall: 0.735
Failure F1: 0.813
```

The project demonstrates an end-to-end predictive-maintenance workflow from engineering data analysis and feature engineering to machine-failure prediction and maintenance risk scoring.
