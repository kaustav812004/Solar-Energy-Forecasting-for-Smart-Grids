A final-year B.Tech major project focused on **24-hour day-ahead solar photovoltaic (PV) power forecasting** using historical PV generation and meteorological data. The project compares a **Persistence baseline**, **Long Short-Term Memory (LSTM)** network, and an attention-based **Transformer** model.

## Project Objective

The objective is to predict the **next 24 hours of PV power output** from the preceding **168 hours (7 days)** of solar and meteorological observations.

The forecasting pipeline is designed to study both prediction accuracy and physically important solar-generation behaviour, including seasonal variability and rapid power ramps.

## Data Sources

The project combines two datasets:

### PVGIS
PVGIS provides PV-system and solar-related variables, including:

- `P` — PV power output (forecast target)
- `G_pv / G(i)` — plane-of-array solar irradiance
- `H_sun` — solar elevation angle
- `T2m_pvgis` — air temperature at 2 m
- `WS10m_pvgis` — wind speed at 10 m
- `Int` — reconstruction/interpolation indicator where applicable

### NASA POWER
NASA POWER provides additional meteorological information:

- `G_allsky` — all-sky surface shortwave radiation
- `G_clrsky` — clear-sky surface shortwave radiation
- `T2m_nasa` — air temperature
- `RH2M` — relative humidity
- `WS10m_nasa` — wind speed
- `WD10M` — wind direction
- `PS` — surface pressure
- `PSC` — corrected/derived pressure-related variable used in the prepared dataset

The datasets are temporally aligned and merged to create a unified time-series dataset for the Delhi study region.

## Physics-Informed Feature Engineering

In addition to the original variables, several derived features are used.

### Clear-Sky Ratio

The clear-sky ratio captures atmospheric/cloud attenuation:

`CSR = G_allsky / G_clrsky`

### Temperature Features

- `T2m_avg` — combined temperature representation
- `T2m_diff` — difference between temperature measurements

### Temporal Features

- Hour
- Month
- Day
- Weekday

Periodic time variables are represented using sine/cosine encodings such as:

`hour_sin`, `hour_cos`, `day_sin`, `day_cos`

These features help the models represent the daily and seasonal periodicity of solar generation.

## Forecasting Pipeline

The main workflow is:

`PVGIS + NASA POWER -> Alignment/Merging -> Cleaning -> Feature Engineering -> Scaling -> Time Split -> Sliding Windows -> Models -> Evaluation`

### Sliding Window

Each model receives the previous **168 hourly observations** and predicts the following **24-hour PV power profile**.

- Input window: `168 hours`
- Forecast horizon: `24 hours`

Conceptually:

`[t-167, ..., t] -> [P(t+1), ..., P(t+24)]`

## Models

### Persistence Baseline

A persistence forecast is used as a reference model. This provides a simple benchmark against which the deep-learning models can be evaluated.

### LSTM

An LSTM network is used to model temporal dependencies in the historical solar and weather sequence.

### Transformer

The Transformer uses self-attention to identify relationships between different time steps in the 168-hour input window. It is evaluated against both LSTM and persistence forecasting.

## Training

The implementation uses Python with TensorFlow/Keras and scikit-learn.

Key elements include:

- Chronological train/validation/test separation
- Feature and target scaling
- Adam optimization
- Validation monitoring
- Early stopping
- Learning-rate reduction

## Model Evaluation

The models are evaluated using multiple complementary analyses rather than a single error metric.

### Forecasting Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Normalized error measures where applicable

Daylight-only evaluation is also used because the large number of zero-power nighttime observations can otherwise make PV forecasting performance appear artificially strong.

### Persistence Skill

Model performance is compared against persistence to determine whether the learned forecast provides meaningful improvement over a simple baseline.

### Predicted vs Actual Analysis

Scatter plots and time-series plots are used to examine:

- Agreement between actual and predicted PV output
- Peak-power bias
- Under/over-prediction
- Overall model fit

### Seasonal Analysis

Forecast errors are studied across Delhi's major seasonal regimes:

- Pre-monsoon
- Monsoon
- Post-monsoon
- Winter

This analysis is useful because cloud cover, monsoon variability, winter haze, aerosols, temperature, and irradiance conditions influence PV generation differently throughout the year.

### Ramp Analysis

PV power ramps are defined as:

`ΔP(t) = P(t+1) - P(t)`

Ramp analysis evaluates how well the models capture rapid changes in generation caused by sunrise/sunset and changing atmospheric or cloud conditions.

### Statistical Validation

Paired statistical testing is used to investigate whether differences in model errors are statistically significant rather than relying only on aggregate MAE/RMSE values.

## Solar-Physics Context

The dissertation connects the forecasting pipeline with the physical mechanisms governing PV generation, including:

- Solar irradiance
- Solar geometry and elevation
- Direct and diffuse radiation
- Clear-sky versus all-sky conditions
- Atmospheric attenuation
- Delhi-specific haze, aerosols, and monsoon cloud variability
- Temperature effects on PV performance
- PV power dependence on irradiance
- Solar power ramp events

The aim is to maintain a clear connection:

`Solar Physics -> Dataset Variables -> Engineered Features -> Forecasting Model -> Performance Analysis`

## Industrial Relevance

Day-ahead PV forecasting can support:

- Grid generation scheduling
- Renewable-energy integration
- Reserve planning
- Battery energy storage scheduling
- Day-ahead electricity-market participation
- Solar-plant operation
- Management of rapid renewable-generation ramps

The prediction itself is therefore not the final objective; it provides information that can support operational and economic decisions.

## Technology Stack

- Python
- NumPy
- Pandas
- scikit-learn
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook
- LaTeX / Overleaf

## Current Project Status

Completed work includes:

- PVGIS and NASA POWER dataset preparation
- Temporal alignment and dataset fusion
- Feature engineering
- Scaling and chronological splitting
- 168-to-24 sliding-window generation
- Persistence baseline
- LSTM forecasting
- Transformer forecasting
- MAE/RMSE-based evaluation
- Daylight evaluation
- Scatter-plot analysis
- Seasonal analysis
- Ramp analysis
- Statistical model comparison
- Dissertation documentation

## Future Work

Possible extensions include:

- Incorporating numerical weather prediction / future weather forecasts
- Probabilistic solar forecasting and uncertainty intervals
- Additional machine-learning baselines
- More rigorous forecast-comparison statistical tests
- Hyperparameter optimization
- Explainability/attention analysis
- Real-time inference pipeline
- Battery scheduling using predicted PV generation
- Validation across multiple geographic locations

## Dissertation

This repository forms the implementation component of the final-year major project dissertation on **Day-Ahead Solar PV Power Forecasting using Deep Learning**.

The dissertation covers the solar-physics background, datasets, preprocessing, feature engineering, LSTM and Transformer architectures, experimental methodology, model evaluation, seasonal behaviour, ramp forecasting, statistical validation, industrial relevance, limitations, and future research directions.

## Authors

**Kaustav Ghosh**  
B.Tech — Information Technology  
Guru Gobind Singh Indraprastha University (GGSIPU)
"""

path = "/mnt/data/README.md"
pypandoc.convert_text(readme, 'md', format='md', outputfile=path, extra_args=['--standalone'])
print(f"Created {path}")
