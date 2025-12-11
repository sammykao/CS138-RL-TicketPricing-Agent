# Demand Modeling

This folder contains the demand model training pipeline that learns to predict ticket sale probabilities from historical data.

## Contents

- **`data_extractor.py`**: Extracts and aggregates sales data from the database. Creates training datasets by:
  - Aggregating sales by event, quality tier, and time bin
  - Computing empirical sale probabilities
  - Handling exposure (inventory remaining) calculations

- **`feature_engineer.py`**: Constructs the 21-dimensional feature vector from raw sales data. Features include:
  - Time-to-event (continuous)
  - Price features (relative price, log-relative price, thresholds)
  - Quality score
  - Context features (weekend, playoff, day of week)
  - Interaction terms (price-time, price-urgency, etc.)

- **`demand_fitter.py`**: Fits logistic regression model to predict sale probabilities. Handles:
  - Model training with L2 regularization
  - Platt scaling for probability calibration
  - Cross-validation for hyperparameter tuning

- **`model_serializer.py`**: Saves and loads trained demand models (`.pkl` files).

- **`model_validator.py`**: Validates model performance using metrics like ROC-AUC, Brier score, and calibration error.

- **`train_model.py`**: Main script to train the demand model end-to-end.

## Usage

To train a new demand model:
```bash
cd learning_environment/demand_modeling
python train_model.py
```

The trained model is saved to `learning_environment/models/` and used by the RL environment to compute sale probabilities.

