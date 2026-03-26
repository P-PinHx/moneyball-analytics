# moneyball-analytics
Moneyball Analytics Project - Predict the number of games a Major League Baseball team will win in a season based on their performance statistics. This is a regression problem using historical MLB team data to build a model that can accurately forecast seasonal win totals.

---

# DataSet

The Data
The dataset contains comprehensive team statistics from the 2016 Lahman Baseball Database, including:

Batting statistics: Runs, hits, home runs, strikeouts, etc. Pitching statistics: Earned run average, saves, strikeouts, etc. Fielding statistics: Errors, double plays, fielding percentage Team information: Year, team name, franchise ID Game outcomes: Wins, losses, championships

 ## MLB Season Wins Prediction: Data Description
 ### Files

  data.csv - Historical MLB team seasons with all features and the target variable (W) - 1698 samples
  predict.csv - MLB team seasons for prediction submission (without the W column) - 567 samples
  sample_submission.csv - A sample submission file in the correct format.

 ### Analysis Support Fields

 These fields are included to support data exploration and analysis but use caution when including them as features in  model training.
```
yearID - The season year (integer, e.g., 2019)
year_label - Categorical label for historical baseball eras (1-8):
1: Pre-1920 (Dead-ball era)
2: 1920-1941 (Live-ball era)
3: 1942-1945 (WWII era)
4: 1946-1962 (Post-war era)
5: 1963-1976 (Pitcher's era)
6: 1977-1992 (Free agency era)
7: 1993-2009 (Steroid era)
8: 2010-present (Post-steroid/analytics era)
```
decade_label - The starting year of the decade (e.g., 2010 for 2010-2019)
win_bins - Categorical binning of win totals:
```
0: < 50 wins
1: 50-69 wins
2: 70-89 wins
3: 90-109 wins
4: 110+ wins
```
Note for Model Training: The above fields are only available in the training data but are not included in test.csv as they would cause data leakage or are derived from the target variable.

###Target Variable
```
W - Number of wins in the season (integer, range ~40-120)
Features
Basic Statistics
G - Games played (integer)
```
###Battle Statistcs
```
R - Runs scored (integer)
AB - At bats (integer)
H - Hits (integer)RA - Runs allowed (integer)
ER - Earned runs allowed (integer)
ERA - Earned run average (float)
CG - Complete games (integer)
SHO - Shutouts (integer)
SV - Saves (integer)
IPouts - Outs pitched (innings pitched × 3) (integer)
HA - Hits allowed (integer)
HRA - Home runs allowed (integer)
BBA - Walks allowed (integer)
SOA - Strikeouts by pitchers (integer)
E - Errors (integer)
DP - Double plays (integer)
FP - Fielding percentage (float)
2B - Doubles (integer)
3B - Triples (integer)
HR - Home runs (integer)
BB - Walks (integer)
SO - Strikeouts (integer)
SB - Stolen bases (integer)
```
###Pitching/Defense Statistics
```
RA - Runs allowed (integer)
ER - Earned runs allowed (integer)
ERA - Earned run average (float)
CG - Complete games (integer)
SHO - Shutouts (integer)
SV - Saves (integer)
IPouts - Outs pitched (innings pitched × 3) (integer)
HA - Hits allowed (integer)
HRA - Home runs allowed (integer)
BBA - Walks allowed (integer)
SOA - Strikeouts by pitchers (integer)
E - Errors (integer)
DP - Double plays (integer)
FP - Fielding percentage (float)
```
###Derived Features

mlb_rpg - MLB average runs per game for the season (float)

###Era Indicators

Binary indicators for different historical MLB eras:
```
era_1 - Pre-1920: Dead-ball era
era_2 - 1920-1941: Live-ball era
era_3 - 1942-1945: WWII era
era_4 - 1946-1962: Post-war era
era_5 - 1963-1976: Pitcher's era
era_6 - 1977-1992: Free agency era
era_7 - 1993-2009: Steroid era
era_8 - 2010-present: Post-steroid/analytics era
```
###Decade Indicators

Binary indicators for each decade (1910s-2010s):
```
decade_1910 - 1910s
decade_1920 - 1920s
decade_1930 - 1930s
decade_1940 - 1940s
decade_1950 - 1950s
decade_1960 - 1960s
decade_1970 - 1970s
decade_1980 - 1980s
decade_1990 - 1990s
decade_2000 - 2000s
decade_2010 - 2010s
```
---

#Evaluation

Outcome is evaluated using Mean Absolute Error (MAE), which measures the average absolute difference between the predicted wins and actual wins. Lower scores indicate better performance, with a perfect score being 0. The MAE is calculated as the mean of the absolute values of the differences between predicted and actual wins across all teams.

---



