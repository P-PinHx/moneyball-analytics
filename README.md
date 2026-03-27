# ⚾ Moneyball Analytics – Predicting Baseball Wins with Machine Learning

Predicting team wins using sabermetrics + machine learning.

This project builds a stacked ensemble model that predicts baseball team wins using historical statistics. The approach combines traditional baseball analytics principles with modern ML stacking techniques.

# 🏆 Best Kaggle Score:
2.90946 MAE

## Competition:
NTU DSAI DS2F Moneyball Analytics
---

# Project Motivation
The objective is to predict the number of wins (W) for baseball teams using historical statistics.

The project is inspired by the Moneyball philosophy, which uses data to uncover winning strategies rather than relying on intuition.

### The fundamental idea:
```
Winning baseball teams score more runs than they allow.
```
This insight is strongly supported by sabermetric research.

# Baseball Analytics Background
In baseball, a team scores runs by hitting and advancing around bases until reaching home plate.
A game consists of nine innings, where teams alternate between offense and defense.
Runs are scored when a player completes the circuit of bases and returns to home plate.

### Therefore:
```
Wins ≈ Offensive production − Defensive prevention
```
This is why run differential is the most important statistic in baseball analytics.

---

# Key Sabermetric Principle
### Run Differential
```
run_diff = R - RA
```
Where:
```
| Variable | Meaning      |
| -------- | ------------ |
| R        | Runs scored  |
| RA       | Runs allowed |
```
### Run Differential per Game
```
run_diff_pg = (R - RA) / G
```
This normalizes team performance across seasons with different numbers of games.

---

# Dataset

Two datasets are used.

## Training Data
```
data.csv
```
Contains historical team statistics and wins.

## Prediction Data
```
predict.csv
```
Used to generate Kaggle leaderboard submissions.
The starter notebook loads both datasets and prepares features before training a regression model.

---

# Model Architecture

The final system uses stacked ensemble learning.
```
                 Feature Engineering
                        │
                        ▼
                 Base Models
      ┌─────────────┬─────────────┬─────────────┬─────────────┐
      │      ElasticNet     │        Ridge        │     RandomForest   │    GradientBoost   │
      └─────────────┴─────────────┴─────────────┴─────────────┘
                        │
                        ▼
                Stacking Regressor
                 Linear Regression
                        │
                        ▼
                Final Wins Prediction
```
---

# Base Models

## ElasticNet Regression
Captures linear relationships while handling correlated baseball statistics.

Example correlations:
```
R ↔ H ↔ HR ↔ BB
RA ↔ ERA ↔ ER
```
ElasticNet helps control multicollinearity.
---

# Ridge Regression

Stabilizes predictions when many variables are correlated.

## Works well with:
```
- hitting statistics
- pitching metrics
- defensive indicators
```
## Random Forest

Captures non-linear interactions.

Examples:
```
1) strong offense + weak defense
2) ballpark factor + power hitting
```
## Gradient Boosting
Learns subtle statistical patterns.
Useful for small structured datasets like baseball season statistics.

---

# Meta Model
A Linear Regression stacker combines predictions from all base models.

## Model contribution:
```
| Model         | Weight |
| ------------- | ------ |
| ElasticNet    | 0.457  |
| Ridge         | 0.343  |
| RandomForest  | -0.111 |
| GradientBoost | 0.303  |
```
## Interpretation:
Most predictive power comes from linear relationships, confirming the sabermetric hypothesis.
```
> ElasticNet adds small signal
> Ridge is doing most of the work
> RandomForest may be hurting performance
> GradientBoost adds signal
```
---

# Feature Engineering
Two engineered features significantly improved performance.

### Run Differential
```
run_diff = R - RA
```
### Run Differential Per Game
```
run_diff_pg = run_diff / G
```
These features encode the fundamental relationship between offense and defense.

---

# Feature List
Key statistics include:

## Offensive Metrics
```
Runs (R)
Hits (H)
Home Runs (HR)
Walks (BB)
Strikeouts (SO)
```
## Pitching Metrics
```
Runs Allowed (RA)
Earned Run Average (ERA)
Strikeouts Allowed (SOA)
```
## Defensive Metrics
```
Errors (E)
Double Plays (DP)
Fielding Percentage (FP)
```
## Contextual Metrics
```
Ballpark factors (BPF)
Pitching park factor (PPF)
Attendance
```
These contextual variables account for environmental effects on scoring.

---

# Model Pipeline
The pipeline follows these steps:

## 1️⃣ Load datasets
```
data.csv
predict.csv
```
## 2️⃣ Feature engineering

Add sabermetric predictors.
```
run_diff
run_diff_pg
```
## 3️⃣ Train base models
```
ElasticNetCV
RidgeCV
RandomForestRegressor
GradientBoostingRegressor
```
## 4️⃣ Train stacking model
```
StackingRegressor
```
## 5️⃣ Generate predictions
Predicted wins are rounded and exported.
```
submission_predict.csv
```
---

# Repository Structure
```
moneyball-analytics/
│
├── data/
│   ├── data.csv
│   └── predict.csv
│
├── src/
│   └── stacking_model.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── submission_predict.csv
│
└── README.md
```
---

# Performance
```
| Model               | Kaggle Score |
| ------------------- | ------------ |
| ElasticNet baseline | ~3.06        |
| Stacking ensemble   | ~3.00        |
| Final tuned stack   | **2.90946**  |
```
---

# Key Insights

## 1️⃣ Baseball Wins Are Mostly Linear
Despite complex interactions, linear models explain most variance.

## 2️⃣ Run Differential Is the Best Predictor
```
run_diff = R - RA
```
captures team dominance.

## 3️⃣ Stacking Improves Stability
Combining models reduces overfitting and prediction variance.

---

# Technologies Used
```
- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
```

# Future Improvements
Potential enhancements:
```
- XGBoost / LightGBM stacking
- Pythagorean win expectation
- Bayesian regression
- era-adjusted scoring metrics
- advanced sabermetrics (WAR, OPS)
```
# References
1) Beginner baseball concepts such as scoring runs, winnings, and base running were referenced from the "Beginner’s Guide to Baseball" document.

2) Starter modeling workflow derived from the competition starter notebook.


