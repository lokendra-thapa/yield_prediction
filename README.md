# Crop Yield Prediction

A Flask-based crop yield prediction project that loads a trained machine learning model, applies preprocessing, and 
serves a web UI for estimating crop yield based on farm attributes.

## Project Overview

This repository includes:
- a dataset used for training and analysis (`dataset_for_yield_prediction.csv`)
- a model training and analysis script (`yield_prediction_analysis.py`)
- a Flask web application for predictions (`app.py`)
- saved preprocessing objects and trained model artifacts
- a user interface to submit farm inputs and view predicted yield

## Features

- Loads a pre-trained regression model from `yield_prediction_model.joblib`
- Uses feature scaling and label encoding stored in `scaler.joblib` and `label_encoders.joblib`
- Predicts crop yield based on:
  - crop type
  - farm area (acres)
  - irrigation type
  - fertilizer used (tons)
  - pesticide used (kg)
  - soil type
  - season
  - water usage (cubic meters)
- Calculates a simple confidence interval using dataset standard deviation
- Provides a nice Bootstrap-based dashboard for input and result display

## Files

- `app.py` - Flask web app that serves the UI and prediction API
- `yield_prediction_analysis.py` - data preprocessing, model training, evaluation, and plot generation script
- `dataset_for_yield_prediction.csv` - source dataset used for analysis and training
- `yield_prediction_model.joblib` - saved trained regression model
- `scaler.joblib` - saved feature scaler
- `label_encoders.joblib` - saved label encoders for categorical features
- `templates/index.html` - front-end UI template for the Flask app
- `static/` - static assets, including generated plot image output
- `requirements.txt` - Python dependencies
- `label_encoder.joblib` - additional saved encoder artifact (unused by app but present in repo)

## Dataset Schema

The app expects and the training script uses the following dataset columns:
- `Farm_ID`
- `Crop_Type`
- `Farm_Area(acres)`
- `Irrigation_Type`
- `Fertilizer_Used(tons)`
- `Pesticide_Used(kg)`
- `Soil_Type`
- `Season`
- `Water_Usage(cubic meters)`
- `Yield(tons)`

Derived features created in training and prediction:
- `Fertilizer_per_acre`
- `Pesticide_per_acre`
- `Water_per_acre`



## Prediction API

The app exposes a `/predict` endpoint that accepts JSON input with these fields:
- `crop_type`
- `farm_area`
- `irrigation_type`
- `fertilizer`
- `pesticide`
- `soil_type`
- `season`
- `water_usage`

Example request body:
```json
{
  "crop_type": "Wheat",
  "farm_area": 10,
  "irrigation_type": "Drip",
  "fertilizer": 1.5,
  "pesticide": 0.8,
  "soil_type": "Loamy",
  "season": "Summer",
  "water_usage": 2000
}
```

Example response:
```json
{
  "success": true,
  "prediction": 23.45,
  "confidence_interval": {
    "lower": 19.76,
    "upper": 27.14
  }
}
```

## Model Training and Analysis

The training script `yield_prediction_analysis.py`:
- loads `dataset_for_yield_prediction.csv`
- label-encodes categorical features
- creates derived per-acre features
- splits data into training/testing sets
- scales features with `StandardScaler`
- trains `RandomForestRegressor` and `GradientBoostingRegressor`
- evaluates models with MSE, R2, MAE and cross-validation
- saves the best model and preprocessing objects to `.joblib` files
- generates analysis plots in `static/analysis_plots.png`


## Dependencies

The project depends on:
- Python
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install these via `pip install -r requirements.txt`.
