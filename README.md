# Sydney Weather Intelligence — Experimentation

Machine learning experimentation for forecasting two weather-derived indices for
Sydney, Australia, using historical observations from the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).

**Abhishek Chopda** — Student ID **25611094**
36120 Advanced Machine Learning Application — Assessment Task 2

---

## The project

Two prediction targets are derived from daily weather for Sydney
(latitude −33.8688, longitude 151.2093) and forecast by separate models:

| Target | Type | Horizon | Output |
|---|---|---|---|
| Climate Comfort Index (CCI) | Regression | +1, +2 and +3 days | Score 0–100 (higher is more comfortable) |
| Weather Hazard Category (WHC) | Multiclass classification | Exactly +7 days | Low, Moderate, High or Extreme Risk |

Four experiments were run, two per target. The trained models are served through
a FastAPI application deployed on Render.

## Repositories and links

| Item | Link |
|---|---|
| Experimentation (this repo) | <https://github.com/abychopda/sydney-weather-intelligence> |
| FastAPI service | <https://github.com/abychopda/weather-intelligence-api> |
| Reusable Python package | <https://github.com/abychopda/mlkit-aby> |
| Package on TestPyPI | <https://test.pypi.org/project/mlkit-aby/> |
| **Live API** | <https://weather-intelligence-api-s6rd.onrender.com> |
| **Interactive API docs** | <https://weather-intelligence-api-s6rd.onrender.com/docs> |

The Render service runs on the free tier, which sleeps after 15 minutes without
traffic. The first request after that takes about a minute while the service
wakes up; later requests are fast.

## Results at a glance

### Climate Comfort Index — works, and is deployed

| Model | Test MAE (+1 / +2 / +3 days) | Test R² |
|---|---|---|
| **Ridge regression (experiment 1)** | **6.49 / 7.35 / 7.50** | 0.45 / 0.29 / 0.24 |
| Random Forest (experiment 2) | 6.57 / 7.63 / 7.69 | 0.44 / 0.26 / 0.23 |
| Persistence baseline | 7.56 / 9.09 / 9.59 | — |
| Seasonal average baseline | 8.72 / 8.72 / 8.69 | — |

A typical forecast lands within about 6.5 points of the truth one day ahead,
which is well inside the 20-point width of a comfort band, and the correct
comfort band is predicted on 65% of days. Both baselines are beaten clearly.

### Weather Hazard Category — does not work, and is documented as such

| Model | Test macro F1 | Test ROC AUC | High Risk recall |
|---|---|---|---|
| Logistic regression (experiment 1) | 0.24 | 0.38 | 0.00 |
| Random Forest (experiment 2) | 0.26 | 0.46 | 0.00 |

Both models rank hazardous days **no better than chance** (ROC AUC below 0.5),
and a threshold sweep shows precision equal to the base rate at every setting.
The cause is the seven-day horizon: today's weather correlates about 0.06 with
the hazard index a week later, and the strongest single feature explains roughly
2% of the variation. The hazard endpoint is therefore published as experimental
and is not suitable as a safety input.

## Which models are deployed

Both deployed models come from the **experiment 1** notebooks. Experiments 2
tested more complex models and neither improved on its baseline, so neither
saves a model artefact — their value is the evidence for that conclusion.

| Artefact | Produced by | Serves |
|---|---|---|
| `models/comfort_climate/cci_ridge_h1.joblib` | CCI experiment 1 | `/predict/index/comfort_climate` |
| `models/comfort_climate/cci_ridge_h2.joblib` | CCI experiment 1 | same |
| `models/comfort_climate/cci_ridge_h3.joblib` | CCI experiment 1 | same |
| `models/weather_hazard/whc_logistic_h7.joblib` | WHC experiment 1 | `/predict/category/weather_hazard` |

Each artefact stores the fitted pipeline, the feature recipe, the exact feature
order, how many days of history a prediction needs, and the measured test
metrics — so the API can rebuild identical features for any new date.

## The notebooks

| Notebook | Outcome |
|---|---|
| `notebooks/comfort_climate/25611094-cci-experiment-1-linear-baseline.ipynb` | Partially confirmed — **produced the deployed CCI models** |
| `notebooks/comfort_climate/25611094-cci-experiment-2-random-forest.ipynb` | Rejected — tree models did not beat Ridge |
| `notebooks/weather_hazard/25611094-whc-experiment-1-logistic-baseline.ipynb` | Rejected — no usable signal at seven days |
| `notebooks/weather_hazard/25611094-whc-experiment-2-random-forest.ipynb` | Rejected — better than experiment 1, still below chance |

Each notebook follows the provided experiment template and is self-contained:
it loads the raw data, builds its target, explores the data, selects and
engineers features, splits chronologically, trains and evaluates, and ends with
a stated outcome.

`reports/NOTES.md` records the decisions taken across all four experiments, the
measurements behind them, and the ideas that were tested and rejected.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1. Install uv if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install the project and its dependencies
uv sync

# 3. Download the raw weather data (required before running any notebook)
uv run python -m weather_intelligence.dataset

# 4. Launch Jupyter
uv run jupyter lab
```

Step 3 retrieves 227,928 hourly observations for Sydney from 2000 to 2025 and
writes them to `data/raw/`. The raw data is not committed to version control, so
this step must be run once before the notebooks will execute. It needs an
internet connection and takes about a minute.

Notebooks should be run top to bottom. The experiment 2 notebooks each take
about five minutes because of their hyperparameter searches.

## The `mlkit-aby` package

Reusable functionality lives in a separate package, extended from the one built
for AT1 and published to TestPyPI. The notebooks call its functions one at a
time so that each step is visible rather than hidden behind a single wrapper.

| Module | Provides |
|---|---|
| `helpers.weather` | Open-Meteo download and hourly-to-daily aggregation |
| `helpers.targets` | The CCI and WHI formulas and the WHC bands |
| `helpers.timeseries` | Lag, rolling and cyclical calendar features |
| `helpers.features` | Recipe-driven feature assembly and rainfall features |
| `helpers.modeling` | Model training and evaluation helpers |
| `helpers.importance` | Coefficient, impurity and permutation importance |
| `helpers.eda`, `helpers.visualization` | Summaries and plots |

The package is installed from TestPyPI automatically by `uv sync`.

## Project organisation

```text
├── data
│   ├── raw             <- Hourly observations from Open-Meteo (not committed)
│   ├── interim         <- Intermediate data
│   └── processed       <- Final modelling datasets, saved by the notebooks
│
├── models
│   ├── comfort_climate <- Deployed CCI artefacts (3 horizons)
│   └── weather_hazard  <- Deployed WHC artefact
│
├── notebooks
│   ├── comfort_climate <- CCI experiments 1 and 2
│   └── weather_hazard  <- WHC experiments 1 and 2
│
├── reports
│   ├── NOTES.md        <- Decisions, measurements and rejected ideas
│   └── figures         <- Generated figures
│
├── weather_intelligence <- Project source code and configuration
│
├── pyproject.toml
└── README.md
```

## Notes on method

- **Chronological splits only.** Weather is highly autocorrelated, so a random
  split would place near-identical adjacent days on both sides of the boundary
  and inflate the score. The last few days of each training period are also
  removed, because their targets fall inside the following period.
- **No future information in any feature.** Lag and rolling features shift
  backwards only, and the single forward shift in each notebook is the one that
  creates the target.
- **Test sets used once.** All tuning is done on cross-validation and a separate
  validation period. In both experiment 2 notebooks the model that won on
  validation lost on test, which is the clearest argument for holding the test
  set back.
- **The models assume a prediction is made once the day is complete**, because
  the features include that day's full 24 hours of weather.
