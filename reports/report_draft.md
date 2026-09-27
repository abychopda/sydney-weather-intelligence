# Assignment 2 — ML as a Service

**Abhishek Chopda** · Student ID 25611094 · 2026 Spring
GitHub username: abychopda

Experiment repo: https://github.com/abychopda/sydney-weather-intelligence
Package repo: https://github.com/abychopda/mlkit-aby
API repo: https://github.com/abychopda/weather-intelligence-api
Render: https://weather-intelligence-api-s6rd.onrender.com

36120 Advanced Machine Learning Application · Master of Data Science and Innovation · University of Technology Sydney

---

## 1. Executive Summary

Open Meteo is extending its weather platform with machine learning services. This project develops, evaluates and deploys two predictive services for Sydney, built from historical weather observations and delivered through a REST API.

Two targets were derived from 227,928 hourly observations covering 2000 to 2025. The Climate Comfort Index (CCI) condenses temperature, humidity, wind, cloud cover and rainfall into a 0–100 score describing how suitable a day is for outdoor activity, forecast one, two and three days ahead. The Weather Hazard Category (WHC) rates conditions as Low, Moderate, High or Extreme Risk, forecast exactly seven days ahead.

Four experiments were conducted, two per target, comparing linear models against tree-based ensembles with chronological splitting, hyperparameter tuning and feature selection at each stage.

The comfort service works. A ridge regression model predicts the CCI with a mean absolute error of 6.49 points one day ahead, rising to 7.50 at three days, and places the forecast in the correct comfort band on 65% of test days. It clearly outperforms both a persistence baseline (7.56) and a seasonal average baseline (8.72). A Random Forest and a gradient boosting model were tested and neither improved on it.

The hazard service does not work, and this is reported as a finding rather than concealed. Both models rank hazardous days no better than chance, with ROC AUC scores of 0.38 and 0.46 against the 0.50 that random ordering achieves. The cause is the seven-day horizon: today's weather correlates approximately 0.06 with the hazard index a week later, and the strongest available feature explains around 2% of its variation. No algorithm, feature set, class weighting or decision threshold overcame this.

Both models are deployed on Render as a Dockerised FastAPI service exposing five endpoints, with the hazard endpoint documented as experimental and unsuitable for safety decisions.

---

## 2. Business Understanding

### 2a. Business Use Cases

Open Meteo's platform publishes raw weather variables, which are accurate and complete but place the burden of interpretation on the user. Someone planning an outdoor event must read five separate forecasts and combine them into a judgement about whether Saturday is suitable. The opportunity is to publish interpretation rather than measurement: a single forward-looking score, calculated consistently and available through an API, lets applications act on one number instead of five. This project addresses two such use cases.

The first is outdoor activity planning. The Climate Comfort Index combines five weather variables into a 0–100 score describing how pleasant a day will be outdoors, forecast one to three days ahead. Outdoor venues, tour operators, event organisers and consumer weather applications are the intended consumers. The commercial value lies in committing earlier: staff can be rostered, stock ordered and marketing scheduled against a forward estimate rather than a guess.

The second is operational risk planning. The Weather Hazard Category rates the severity of conditions across four levels, forecast exactly seven days ahead. Councils, transport operators and utilities are the intended consumers, and the seven-day horizon is the source of the value: a week's notice allows work to be rescheduled and crews repositioned, where a one-day warning only allows people to take shelter.

### 2b. Key Objectives

The primary objective is to produce forecasts accurate enough to change a decision, and to be explicit about when they are not.

Three groups of stakeholders have distinct requirements. Developers integrating the endpoints require stable response formats, sensible handling of invalid input, and published performance figures so they can judge how far to rely on a prediction. The organisations acting on the forecasts require accuracy sufficient for planning and honesty about uncertainty at longer horizons. The public receive advice derived from these predictions without ever interacting with the service.

The two services carry different risk profiles. Comfort predictions are advisory: an incorrect forecast costs money and goodwill, and over-prediction is the more damaging direction because customers are brought outdoors into unpleasant conditions. Hazard predictions may inform safety decisions, where a missed warning is considerably more costly than a false alarm.

A further objective follows from that asymmetry. Where a service cannot meet the standard its use case requires, the project must establish and publish that limitation rather than deploy a model whose confidence is unwarranted. This objective proved decisive for the hazard service.

---

## 3. Data Understanding

All data comes from the Open-Meteo Historical Weather API (Open-Meteo, n.d.), retrieved for Sydney (latitude −33.8688, longitude 151.2093). The archive endpoint was queried for 2000-01-01 to 2025-12-31, returning **227,928 hourly observations** across seven variables: temperature, relative humidity, wind speed, cloud cover, precipitation, wind gusts and snowfall. Data from 2026 onwards was excluded from all experimentation and reserved as production data.

Requests specify the `Australia/Sydney` timezone. This is not cosmetic: the targets are defined on calendar days, and UTC timestamps would shift each Sydney day by ten or eleven hours, mixing observations from two local days into one aggregate.

The dataset is complete across 9,497 days, with no missing values, date gaps or duplicates. This reflects the source: Open-Meteo's archive is a reanalysis product, in which observations are assimilated into a physical model to produce a gap-free reconstruction. No imputation was required, though the corresponding limitation is that these are modelled estimates rather than direct station readings.

**Deriving the targets.** Neither target exists in the source data. Hourly observations are first aggregated to one value per day, then the specified formulas are applied. The order matters, because the formulas use MIN, MAX and ABS and give different results if applied hourly and averaged afterwards. The constants confirm the intended reading: precipitation is divided by 20, a plausible daily rainfall total in millimetres but almost never an hourly rate. This was confirmed by the subject coordinator.

The aggregation chosen for each variable reflects what it represents. Temperature, humidity, wind speed and cloud cover use the daily mean, because comfort describes typical conditions. Rainfall and snowfall use the daily total. Wind gusts use the daily maximum, because a hazard is defined by the worst moment of the day: a single 95 km/h gust brings down a tree regardless of how calm the remaining hours were.

**[FIGURE 1: CCI distribution and mean CCI by month]**

The Climate Comfort Index is approximately normal, centred on 72.6 with a standard deviation of 10.2 and ranging from 28.6 to 98.9. Sydney's mild climate shows clearly: 66% of days fall in the "comfortable" band and 23% in "excellent", while no day in 26 years scores below 20. Comfort peaks in the shoulder seasons and falls in mid-winter. A slow downward drift is present, with mean CCI falling from 73.9 in the 2000s to 69.9 in the 2020s.

**[FIGURE 2: Weather Hazard Category class distribution, log scale]**

**[TABLE 1]**

| Category | Days | Share | Frequency |
|---|---|---|---|
| Low Risk | 5,242 | 55.2% | — |
| Moderate Risk | 3,970 | 41.8% | — |
| High Risk | 274 | 2.9% | about 11 per year |
| Extreme Risk | 11 | 0.12% | about one every 2.4 years |

Two consequences follow. Accuracy is unusable as a metric, since predicting Low Risk every day scores 55% while never warning about anything. And Extreme Risk cannot realistically be learned from eleven examples, so it is treated throughout as a documented limitation rather than a modelling target.

A further property affects this target specifically: **snowfall is zero on every one of the 9,497 days**. Ten per cent of the specified index weight is inert for this location, the maximum attainable score is 90 rather than 100, and the highest value observed is 83.1.

---

## 4. Data Preparation

**Cleaning and validation.** The daily dataset required no cleaning, but this was established by checking rather than assumed. No missing values were found, which matters because pandas skips missing values when aggregating, so a day built from six hourly readings would produce a mean indistinguishable from one built from twenty-four. Date continuity was confirmed across all 9,497 days, which is what makes positional shifting safe: lag and rolling features shift *rows* rather than dates, so one missing day would silently turn "one day ago" into two, and shift the hazard horizon from seven days to eight, raising no exception. All variables fell within physically plausible bounds and both indices within their defined 0–100 range, which for code-generated targets checks the formulas directly.

No outliers were removed: the largest values, 138 mm of rain and a 102 km/h gust from the February 2020 east coast low, are real events and precisely the days the hazard service exists to identify.

**Feature engineering.** A single day's weather is a poor basis for forecasting, so five families of features were derived. *Lags* copy values from 1, 2, 3 and 7 days earlier onto the current row, giving the recent trajectory rather than a snapshot. *Rolling means and standard deviations* describe the prevailing pattern and how settled it has been. *Cyclical calendar features* encode day-of-year and month as sine and cosine pairs, so 31 December and 1 January sit adjacent rather than 364 units apart. *Within-day features* recover variation that daily means discard, such as the temperature range and hours of rain. *Rainfall features* describe rain as a pattern — a wet-day flag, rolling rain-day counts, rolling totals and the current dry spell — because roughly half of all days record no rain, making raw millimetres a weak input.

The comfort model uses 3, 7 and 30-day windows; the hazard model uses 7, 14, 30 and 60, because at a seven-day horizon only slower-moving information survives. The 60-day choice was tested rather than assumed: windows from 7 to 180 days were compared and correlation peaks at 60 before declining, since longer averages blur the seasonal pattern that gives these features their value (Appendix Figure A1).

**[TABLE 2]**

| | CCI exp 1 | CCI exp 2 | WHC exp 1 | WHC exp 2 |
|---|---|---|---|---|
| Features built | 92 | 96 | 116 | 123 |
| Features used | 92 | 96 | 116 | 91 |
| Training rows | 2,921 | 2,919 | 6,509 | 6,509 |
| Validation rows | 546 | 544 | 1,454 | 1,454 |
| Test rows | 548 | 546 | 1,454 | 1,454 |

**Feature selection.** Selection was applied at two levels. Among the raw weather variables, wind gusts were dropped from the comfort model as a duplicate of wind speed, while snowfall was retained for the hazard target as a required variable and later confirmed to have an importance of exactly zero. Among the engineered features, near-duplicates were removed using a correlation threshold and the remainder ranked by Random Forest importance, with each candidate set then *measured* on the validation set rather than assumed better. For the comfort model, removing six correlated features worsened validation MAE from 6.198 to 6.235 and reducing to the top 20 worsened it to 6.402, so all 96 were retained. For the hazard model at a stricter 0.85 threshold, 32 features were removed and the reduced set of 91 scored marginally better, so the reduction was kept.

**Splitting, scaling and imbalance.** All splits are chronological. Weather is strongly autocorrelated, so a random split would place near-identical adjacent days on both sides of the boundary, inflating the test score before the model failed on genuinely unseen dates. The comfort model trains on 2015 onwards with validation from January 2023 and testing from July 2024; the hazard model uses the full record with validation from 2018 and testing from 2022. At each boundary the final rows are removed because a row's target lies in its future, so comfort training ends three days before validation begins and hazard training seven.

Scaling was applied only where the algorithm requires it: inside a pipeline for ridge and logistic regression, and not at all for tree models. Class imbalance was handled with class weights rather than resampling, since oversampling methods such as SMOTE fabricate days that never occurred.

---

## 5. Modeling

**Algorithm selection.** Ridge regression (Hoerl & Kennard, 1970) was the comfort baseline. The Climate Comfort Index is a weighted sum of five component scores, four of which change smoothly with their weather variable across the range Sydney experiences, so a linear relationship is a reasonable first hypothesis. Ridge was preferred to ordinary least squares because lagged and rolling versions of the same variable overlap heavily, and ordinary least squares responds by distributing large offsetting coefficients across near-duplicates, while the L2 penalty shrinks them together. Multinomial logistic regression was the hazard baseline, chosen as the classification counterpart so the two tracks remained comparable, and because it produces class probabilities rather than bare labels.

Random Forests (Breiman, 2001) formed the second experiment for both targets, for different reasons. For comfort, the baseline's known weakness: the rain score falls to zero at 20 mm and then stops changing, which a straight line cannot represent. For hazard, a hazardous day is a *combination* of heavy rain, severe gusts and heavy cloud together, which a linear model can only add rather than express as a conjunction. Gradient boosting was a third comparison, adding no new dependency since it is part of scikit-learn.

**Hyperparameter tuning.** All models were implemented with scikit-learn (Pedregosa et al., 2011). Tuning used `GridSearchCV` with `TimeSeriesSplit` rather than ordinary k-fold cross-validation, which shuffles rows across folds and would train a model on days occurring after those it is scored on. Scoring matched the headline metric for each problem: mean absolute error for the regression, macro F1 for the classification.

**[TABLE 3]**

| Model | Parameters searched | Selected |
|---|---|---|
| Ridge (CCI exp 1) | alpha, 0.01 to 1000 | 100 (+1 day), 1000 (+2, +3 days) |
| Random Forest (CCI exp 2) | max_features, min_samples_leaf, max_depth | 0.3 / 5 / 15 |
| Logistic (WHC exp 1) | C, class_weight | C = 0.1, no weighting |
| Random Forest (WHC exp 2) | max_features, min_samples_leaf, class_weight | sqrt / 5 / balanced |
| Gradient boosting (WHC exp 2) | learning_rate, max_leaf_nodes, class_weight | 0.05 / 31 / balanced |

For the Random Forests, `max_features` was the most consequential parameter: with around a hundred heavily overlapping features, allowing every tree to consider all of them makes the trees near-identical, and a forest gains accuracy from its trees disagreeing. The number of trees was excluded from the search and fixed at 200 during tuning, then raised to 400 for the final fit. For the hazard target, `class_weight` was tuned rather than assumed, and weighting by inverse frequency scored best at 0.3205 mean macro F1 against 0.2987 for no weighting.

**Additional techniques.** Three were tested: predicting the *change* from today's comfort index rather than its level, weighting training rows by recency to counter measured drift, and sweeping the hazard decision threshold, since a default classifier with 5% positives rarely predicts them. Three separate comfort models were trained, one per horizon, so each could be tuned independently.

**Deployment architecture.** The models are served through a FastAPI application (Ramírez, n.d.), containerised with Docker and deployed on Render as a web service (Render, n.d.). Reusable functionality lives in a Python package, `mlkit-aby`, published to TestPyPI and installed by both the experimentation project and the API. This is what makes the deployment reliable: each saved artefact contains not only the fitted pipeline but the *feature recipe* describing how its inputs were built, the exact feature ordering, and the days of history a prediction requires. At request time the API reads the recipe from the artefact and rebuilds features using the same package code that trained the model, so the two cannot diverge — a failure mode in which a model silently receives incorrectly constructed inputs and returns confident nonsense.

A request validates the supplied date, downloads the required history from Open-Meteo ending on that date and never beyond it, builds features from the stored recipe and returns the prediction. The comfort models need 30 days of history, the hazard model 60. Five endpoints are exposed: documentation, health, the two predictions, and a metadata endpoint publishing each model's algorithm, features, importance and measured performance. Invalid dates return 400, missing parameters 422 and an unreachable weather service 503. Render's free tier sleeps after fifteen minutes of inactivity.

---

## 6. Evaluation

### 6a. Results and Analysis

All models were scored on test periods held back until the end: 546 days from July 2024 to December 2025 for comfort, 1,454 days from 2022 to 2025 for hazard.

**[TABLE 4]**

| Model | MAE +1 | MAE +2 | MAE +3 | R² +1 |
|---|---|---|---|---|
| **Ridge regression (deployed)** | **6.49** | **7.35** | **7.50** | **0.45** |
| Random Forest | 6.57 | 7.63 | 7.69 | 0.44 |
| Gradient boosting | 6.73 | 7.83 | 7.97 | — |
| Ridge + Forest average | 6.44 | 7.43 | 7.56 | — |
| Persistence baseline | 7.56 | 9.09 | 9.59 | 0.25 |
| Seasonal average baseline | 8.72 | 8.72 | 8.69 | 0.05 |

Ridge performs best at every horizon and clearly beats both baselines. A typical one-day forecast lands within 6.5 points of the truth on a 0–100 scale, comfortably inside the 20-point width of a comfort band, and the correct band is predicted on 65.2% of test days, falling to 62.1% at three days.

The baselines fail in opposite ways. Persistence is competitive at one day but collapses beyond it, recording a negative R² at two and three days, meaning it performs worse than predicting the long-run average. The seasonal baseline is flat across horizons and its R² of 0.05 shows the time of year alone explains very little day-to-day variation. The model's value lies in combining two individually weak signals.

**[FIGURE 4: CCI forecast against actual over the test period]**

**[FIGURE 5: predicted against actual, and residuals]**

Accuracy degrades with horizon, reflecting atmospheric predictability rather than a deficiency in the model. RMSE sits well above MAE at one day, 8.56 against 6.49, indicating most days are forecast well and a minority very poorly. Grouping test errors by the rainfall that occurred locates that tail: 5.61 on dry days and on light-rain days, but **16.56 on the 44 days with more than 10 mm**. The cause is structural, since the rain component falls to zero at 20 mm and then stops changing.

That weakness motivated the second experiment, and the result is worth reporting because it was negative. The Random Forest reduced wet-day error to 15.91, but was slightly worse on dry and light-rain days, which account for 92% of the test period, so the small loss outweighed the gain.

A second finding carries more weight. The Random Forest was **better than ridge on both cross-validation and validation** (6.093 and 6.070 against 6.234 and 6.231) and worse on test. With 546 validation days a difference of 0.15 MAE is within noise, and selecting on tuning scores alone would have deployed the weaker model.

**[TABLE 5]**

| Model | Macro F1 | ROC AUC | High Risk recall |
|---|---|---|---|
| Logistic regression (experiment 1) | 0.24 | 0.38 | 0.00 |
| Random Forest (experiment 2) | 0.26 | 0.46 | 0.00 |
| Gradient boosting | 0.26 | 0.50 | 0.00 |
| Random Forest, 2015+ window | 0.25 | 0.41 | 0.00 |

**[FIGURE 7: confusion matrix]**

No model detects hazardous days. Every ROC AUC falls below 0.50, the value random ordering achieves, and none predicts a single High or Extreme Risk day in four years of test data, missing all 73.

A threshold sweep shows this is not a matter of where the decision boundary sits. Lowering it does produce alerts, catching 46 of 62 hazardous validation days at a threshold of 0.10, but precision stays between 4.3% and 5.5% at every setting while 4.26% of days are hazardous anyway. The flagged days are no better than random, and that recall costs 268 alerts per year.

Three comparisons isolate the cause. Refitting experiment 1's logistic regression on experiment 2's improved feature set changed almost nothing, 0.2358 macro F1 against 0.2410, so the difference between experiments came from the model rather than the features. Training on a shorter, more recent window performed *worse*, at 0.41 ROC AUC, which refutes the explanation experiment 1 offered for its own failure: the class mix has shifted, with High Risk doubling in frequency, but truncating the data costs more rare examples than the drift costs relevance. And class weighting improved macro F1 from 0.2987 to 0.3205 while changing nothing about whether a hazardous day was detected.

The cause was measurable before any model was trained. Today's hazard index correlates 0.06 with the index seven days later, and by five days ahead the correlation is already flat, so a seven-day forecast is no better placed than a ten-day one.

**[FIGURE 3: correlation against forecast horizon, both targets]**

### 6b. Business Impact and Benefits

The comfort service delivers measurable value. It places the forecast in the correct comfort band on 65.2% of days one day ahead and 62.1% at three days, and a typical error of 6.5 points is well inside the 20-point width of a band, so most errors do not change the advice given. Against the alternatives a business would otherwise use, it reduces error by 1.07 MAE compared with assuming tomorrow resembles today, and by 2.23 compared with assuming an average day for the time of year. For an outdoor venue or tour operator, this supports committing to staffing and stock decisions two to three days earlier than would otherwise be prudent.

Two limits should be stated. Band agreement barely improves between three days and one, because most additional error stays within the same band. More importantly, the model is substantially worse on wet days, at 16.56 points, and rain is precisely when an operator most needs the forecast.

The hazard service delivers no value in its current form. Because it never predicts a hazardous day, an integrating system would receive Low or Moderate Risk every day while High Risk days occurred roughly every three weeks, a failure mode in which silence is indistinguishable from an all-clear. Lowering the threshold produces alerts on most days while being correct about 4% of the time, which would exhaust operator trust faster than no service at all.

The recommendation is to publish the comfort service with its wet-day caveat documented, and the hazard endpoint as experimental with per-class performance exposed, directing users needing genuine seven-day warnings to official forecasts.

### 6c. Data Privacy and Ethical Concerns

**Data privacy.** This project carries an unusually low privacy burden. The dataset contains meteorological observations for a single point and no personal information of any kind. The API collects none either: requests consist of a date, and no user identity, location or request history is stored. Open-Meteo is credited as the data source in all three repositories.

**Ethical concerns.** The central issue is not privacy but honest disclosure of capability. The hazard model's failure mode is dangerous: rather than producing obviously wrong answers, it silently returns Low or Moderate Risk every day, indistinguishable from a genuine all-clear. Three measures address this: per-class recall is published through the metadata endpoint, the documentation states the model is experimental and unsuitable as a safety input, and users are directed to official forecasts.

A second concern is automation bias: a single number carries an authority that five raw variables do not. A third is that both targets are constructs rather than measurements, encoding a specified definition of comfort — 22 °C as ideal — that is not universal. Outdoor workers, elderly people and people with disabilities experience the same weather very differently.

**Potential impacts and risks for Indigenous people.** The data used here is environmental rather than Indigenous, and Sydney is a well-served metropolitan area, so direct risks are limited. Several indirect considerations deserve attention.

The most significant is distributional. This service works only for Sydney, while the communities most exposed to extreme weather and least served by existing infrastructure include many remote Aboriginal and Torres Strait Islander communities. Building for a major city first is a defensible starting point, but a platform that improves forecasting only where forecasting is already good widens an existing gap rather than closing it.

Should such a service be extended to cover Country, two obligations would follow. Indigenous Data Sovereignty principles hold that data relating to Indigenous lands and communities should be governed by those communities (Maiam nayri Wingara, 2018), so extension would require consent and participation rather than simply widening a bounding box. And Aboriginal seasonal knowledge describes local weather with a richness a four-level hazard band cannot approach: the D'harawal calendar of the Sydney region recognises six seasons keyed to ecological indicators rather than fixed dates (Bureau of Meteorology, n.d.). A model presenting its categories as authoritative risks displacing knowledge that is both older and, for its own purposes, more precise.

Finally, the risk of a non-functioning hazard service falls unevenly. Communities with fewer alternative sources of weather information would be more exposed to a service that appears to work but never warns.

---

## 7. Conclusion

This project built, evaluated and deployed two machine learning services forecasting weather-derived indices for Sydney, supported by a reusable Python package and a containerised API running on Render.

The comfort service achieved its objective. Ridge regression predicts the Climate Comfort Index with a mean absolute error of 6.49 points one day ahead and 7.50 at three days, placing the forecast in the correct comfort band on 65.2% of test days and clearly outperforming both baselines. It is deployed and operational.

The hazard service did not, and establishing that rigorously became the more valuable outcome. Two experiments, four training configurations, three class-weighting strategies, eight thresholds and two training windows all produced models ranking hazardous days no better than chance, for a cause measurable before any model was trained: today's weather correlates 0.06 with the hazard index seven days later.

Four findings generalise beyond this project. Feasibility is largely determined by the forecast horizon, and a correlation analysis at the outset would have predicted both outcomes. In both second experiments the model that won on cross-validation and validation lost on test, the clearest argument for reserving a test set and using it once. Feature engineering cannot create information that is absent, as refitting the hazard model on an improved feature set showed. And a metric can improve while the service does not, as class weighting demonstrated.

**Limitations.** The models cover a single location and do not generalise to other climates. The source is reanalysis output rather than direct station observation. Both targets are specified constructs, so the models predict definitions rather than phenomena. Extreme Risk, at eleven days in twenty-six years, cannot be learned from the available data. The comfort test set covers 546 days, so differences below roughly 0.15 MAE should not be treated as meaningful. The deployment runs on a free tier that sleeps when idle.

**Future work.** For the comfort service, the clearest improvement would address the wet-day weakness, either through a loss function penalising over-prediction more heavily than under-prediction, or through a seasonal climatology feature supplying the normal index value for each day of the year. For the hazard service, further algorithm selection is not the answer; what is needed is different data — synoptic pressure patterns, oceanic indices such as the Southern Oscillation Index, or the output of a physical weather model. The last is what official forecasts already use, and it is the honest explanation for why they can forecast a week ahead when a model built on local observations cannot.

---

## 8. References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Bureau of Meteorology. (n.d.). *Indigenous weather knowledge: D'harawal calendar*. Retrieved September 27, 2026, from http://www.bom.gov.au/iwk/

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics, 12*(1), 55–67. https://doi.org/10.1080/00401706.1970.10488634

Maiam nayri Wingara Indigenous Data Sovereignty Collective. (2018). *Indigenous data sovereignty communiqué*. https://www.maiamnayriwingara.org/

Open-Meteo. (n.d.). *Historical weather API documentation*. Retrieved September 27, 2026, from https://open-meteo.com/en/docs/historical-weather-api

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Ramírez, S. (n.d.). *FastAPI documentation*. Retrieved September 27, 2026, from https://fastapi.tiangolo.com/

Render. (n.d.). *Deploy a web service from a Dockerfile*. Retrieved September 27, 2026, from https://render.com/docs/docker

---

## Appendix

**Figure A1 — Choosing the longest rolling window.** Correlation between a rolling mean of the hazard index and the index seven days ahead, for window lengths from 7 to 180 days. The relationship peaks at 60 days and declines thereafter, because longer averages blur the seasonal pattern that gives these features their value. Sixty days was adopted on this evidence.

**[FIGURE A1]**

**Figure A2 — Correlation between the daily weather variables.** Wind speed and wind gusts correlate at 0.84, measuring the same phenomenon at different statistics. This informed the decision to drop gusts from the comfort model, whose formula uses mean wind speed, while retaining both for the hazard model, where a tree is unaffected by correlated inputs.

**[FIGURE A2]**

**Table A1 — Weather Hazard Category decision threshold sweep (validation set, 1,454 days).**

| Threshold | Alerts raised | Hazards caught | Hazards missed | Recall | Precision | Alerts per year |
|---|---|---|---|---|---|---|
| 0.50 | 0 | 0 | 62 | 0.00 | — | 0 |
| 0.40 | 1 | 0 | 62 | 0.00 | 0.000 | 0.3 |
| 0.30 | 6 | 0 | 62 | 0.00 | 0.000 | 1.5 |
| 0.25 | 51 | 1 | 61 | 0.02 | 0.020 | 12.8 |
| 0.20 | 165 | 9 | 53 | 0.15 | 0.055 | 41.4 |
| 0.15 | 448 | 20 | 42 | 0.32 | 0.045 | 112.5 |
| 0.10 | 1,066 | 46 | 16 | 0.74 | 0.043 | 267.8 |
| 0.05 | 1,429 | 61 | 1 | 0.98 | 0.043 | 359.0 |

Hazardous days make up 4.26% of the validation period. Precision remains at or near that base rate at every threshold, which is the evidence that the flagged days are no better than days selected at random.

**Table A2 — API endpoints and example responses.**

| Endpoint | Example response |
|---|---|
| `GET /health` | `{"status": "healthy", "message": "Weather Intelligence API is running", "models_loaded": 4}` |
| `GET /predict/index/comfort_climate?date=2025-01-01` | `{"input_date": "2025-01-01", "predictions": {"comfort_climate": {"2025-01-02": 82.0, "2025-01-03": 78.7, "2025-01-04": 78.0}}}` |
| `GET /predict/category/weather_hazard?date=2025-01-01` | `{"input_date": "2025-01-01", "predictions": {"weather_hazard": {"2025-01-08": "Moderate Risk"}}}` |
| `GET /model-metadata` | Returns target, algorithm, forecast horizon, feature list, feature importance and test metrics for both models |
| `GET /` | Project title, description, version, GitHub URL and endpoint list |
| Invalid input, e.g. `?date=15-03-2026` | `400` with `{"detail": "Invalid date '15-03-2026'. Please use the format YYYY-MM-DD."}` |

**Table A3 — Summary of the four experiments.**

| Experiment | Model | Hypothesis | Outcome | Headline result |
|---|---|---|---|---|
| CCI 1 | Ridge regression | A regularised linear model can forecast the comfort index usefully at one to three days | Partially confirmed | MAE 6.49 / 7.35 / 7.50; **deployed** |
| CCI 2 | Random Forest, gradient boosting | Non-linear models will beat ridge by handling the clipped rainfall term | Rejected | 6.57 against 6.49; better on validation, worse on test |
| WHC 1 | Multinomial logistic regression | A linear classifier with balanced class weights can forecast hazard at seven days | Rejected | ROC AUC 0.38, below chance; 0 of 73 hazardous days flagged |
| WHC 2 | Random Forest, gradient boosting | A tree can represent the conjunction of conditions defining a hazardous day | Rejected | ROC AUC 0.46, still below chance |
