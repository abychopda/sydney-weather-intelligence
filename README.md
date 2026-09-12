# Sydney Weather Intelligence — Experimentation

Machine learning experimentation for forecasting weather-derived indices for
Sydney, Australia, using historical observations from the Open-Meteo API.

**Abhishek Chopda** — Student ID **25611094**
36120 Advanced Machine Learning Application — Assessment Task 2

## Prediction targets

| Target | Type | Horizon | Range |
|---|---|---|---|
| Climate Comfort Index (CCI) | Regression | +1, +2 and +3 days | 0–100 |
| Weather Hazard Category (WHC) | Multiclass classification | Exactly +7 days | 4 classes |

Both targets are derived from daily weather values for Sydney
(latitude −33.8688, longitude 151.2093).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1. Install uv, if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install the project and its dependencies
uv sync

# 3. Download the raw weather data (required before running any notebook)
uv run python -m weather_intelligence.dataset

# 4. Launch Jupyter
uv run jupyter lab
```

Step 3 retrieves historical observations from the Open-Meteo API and writes
them to `data/raw/`. The raw data is not committed to version control, so
this step must be run once before the notebooks will execute. It requires an
internet connection and takes roughly a minute.

## Project organisation

```text
├── data
│   ├── raw            <- Weather observations downloaded from Open-Meteo
│   ├── interim        <- Intermediate transformed data
│   └── processed      <- Final modelling datasets
│
├── models
│   ├── comfort_climate <- Trained CCI regression artefacts
│   └── weather_hazard  <- Trained WHC classification artefacts
│
├── notebooks
│   ├── comfort_climate <- CCI experiment notebooks
│   └── weather_hazard  <- WHC experiment notebooks
│
├── reports            <- Experiment reports
│   └── figures        <- Generated figures
│
├── weather_intelligence <- Project source code
│
├── pyproject.toml     <- Project configuration and dependencies
└── README.md
```

## Related repositories

| Repository | Purpose |
|---|---|
| [mlkit-aby](https://github.com/abychopda/mlkit-aby) | Reusable ML toolkit, published to TestPyPI |
| [weather-intelligence-api](https://github.com/abychopda/weather-intelligence-api) | FastAPI deployment |
