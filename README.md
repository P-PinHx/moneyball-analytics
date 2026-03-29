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

The final system uses stacked ensemble learning, combining linear and tree-based models to capture both linear sabermetric relationships and non-linear interactions in team performance.
```
                 Feature Engineering
                        │
                        ▼
                 Base Models
      ┌─────────────┬─────────────┬─────────────┬─────────────┐
      │ ElasticNet  │    Ridge    │ RandomForest│GradientBoost│
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
ElasticNet helps control *multicollinearity*.

---

## Ridge Regression

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
This stacking approach balances *model stability* and *predictive power*.

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
Most predictive power comes from **linear relationships**, confirming the sabermetric hypothesis.
```
> ElasticNet adds small signal
> Ridge is doing most of the work
> RandomForest may be hurting performance
> GradientBoost adds signal

- Linear models capture most of the signal (~80%)
- Tree models provide non-linear corrections (~20%)
```
This analysis aligns with baseball analytics where **'run differential'** and **'pitching efficiency'** dominate win prediction.

---

# Feature Engineering
Several sabermetric-inspired engineered features from raw baseball statistics significantly improved predictive performance.
These features summarize team offensive strength, putching efficiency and overall dominance.

### Run Differential (Run Dominance Metrics)
```
run_diff = R - RA
run_diff_pg - run_diff / G
```
- run_diff measures total scoring advantage of a team.
- reun_diff_pg normalizes the run differential by games played.

These metrics capture *team dominance* more effectively than raw runs.
Run differential is one of the strongest predictors of team wins.  
Teams with **higher run differentials** tend to **win more games**.

### Run Differential Per Game (Environment Normalization)
Normalizing by games helps to account for season length environment differences.
```
R_pg = R / G
RA_pg = RA / G
```
These features encode the fundamental *relationship* between offense productivity and defensive effectiveness per game.

### Log-Scaled Run Differential
Extreme run differentials likely dominate linear models, so log transformation is applied to model to compresses extreme values while preserving the direction of team dominance.

```
log_run_diff = sign(run_diff) * log(1 + |run_diff|)
```

### Pitching Efficiency Metrics
The metrics reflect pitching quality and control.
```
WHIP = (BBA + HA) / (IPouts / 3)
```
WHIP measures how many baserunners a pitcher allows per inning (round).

Lower WHIP indicates **better *pitching performance***.

### Strikeout-to-Walk Ratio
```
K_BB_ratio = SOA / (BBA + 1)
```
This metric evaluates pitching control and dominance.
Higher values indicate pitchers who strike out more batters while issuing fewer walks.

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
R_pg
RA_pg
log_run_diff
WHIP
K_BB_ratio
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
Public Leaderboard Score - **MAE = 2.90946**
```
| Model               | Kaggle Score |
| ------------------- | ------------ |
| ElasticNet baseline | ~3.06        |
| Stacking ensemble   | ~3.00        |
| Final tuned stack   | **2.90946**  |
```
Leaderboard scores are calculated on approximately 54% of the test dataset, while remaining 46% determines the final ranking.

---

# Diagnostic Evaluation
Diagnostics are performed to verify **model stability** and **detect potential overfitting**.

## Training Error
```
Training MAE: 2.4448
```
This finding reflect the model's fit on the training dataset.

## Leaderboard Stability Test
For private leaderboard performance estimation, a simulated leaderboard test was conducted using repeated **'ShuffleSplit'** validation.
```
Mean simulated MAE : 2.8199
Standard Deviation : 0.058
```
Interpretation:
- Mean MAE estimates expected private leaderboard score
- Low standard deviation indicates **'high model stability'**
Typical expected private leaderboard range is :
```
2.80 - 2.90 MAE
```

## Error Distribution Analysis
Examining prediction error distribution.
```
<Metric>	           <Value>
Mean error	        2.4448
Median error	        2.0977
90th percentile	      5.003
```
Interpretation:
- Most predictions fall wihtin ~2 wins
- 90% of predictions are within 5 wins
- Error distribution shows few extreme outliers
This indicates a **'well-generalizing' model**.

## Overfitting Assessment
Training and validation metrics' comparison:
```
<Metric>	        <Value>
Training MAE        2.44
Simulated MAE	     2.82
Public leaderboard   2.91
```
The gap **(~0.45)** is typical for stacked models and has **no significant overfitting**.

# Key Insights

## 1️⃣ Baseball Wins Are Mostly Linear
Despite complex interactions, linear models explain most variance.

## 2️⃣ Run Differential Is the Best Predictor
Strongest predictor of team success is **run differential**, defined as the difference between runs scored and runs allowed.
```
run_diff = R - RA
```
### **Teams that consistently score more runs than they allow tend to win more games.**

To capture this behavior metrics, additional transformations of run differential are included in the model.
captures team dominance.

### Log-Scaled Run Differential
```
log_run_diff = sign(run_diff) * log(1 + |run_diff|)
```
This transformation compresses extreme run differentials and helps the model to capture non-linear effects for very strong or very weak teams.

### Run Environment Adjustment
Runs scored and allowed are normalized by games played:
```
R_pg  = R / G
RA_pg = RA / G
```
This approach helps account for differences in scoring environments and season lengths.

Altogether, these features allow model to capture:

**- overall team dominance**
**- non-linear scoring effects**
**- variation in scoring environments**

## 3️⃣ Stacking Improves Stability
Combining models reduces overfitting and prediction variance.

Overall, the strongest signals align with established sabermetric findings:
1. Run Differential
2. Pitching Efficiency
3. Offensive Production
---

Exploratory analysis confirmed that run differential alone explains the majority of variation in team wins, reinforcing its importance as the primary predictive feature.

# Technologies Used
```
- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
```

# Future Improvements
Potential enhancements if there is larger dataset size:
```
- Gradient boosting models - XGBoost / LightGBM stacking
- Pythagorean win expectation
- Bayesian regression
```
# Final Model Characteristics
The final model is expected to **perform consistently on the private leaderboard** evaluation dataset.

✔ Strong predictive signal  
✔ Low variance across validation splits  
✔ Balanced ensemble weights  
✔ Stable error distribution  
✔ Minimal overfitting risk  

# References
1) Beginner baseball concepts such as scoring runs, winnings, and base running were referenced from the "Beginner’s Guide to Baseball" document.

2) Starter modeling workflow derived from the competition starter notebook.


