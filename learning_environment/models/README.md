# Models

This folder contains trained machine learning models.

## Contents

- **`demand_model_v12_anti_gouge.pkl`**: Trained logistic regression demand model. This model predicts the probability that a ticket will sell before the event, given:
  - Current price (relative to initial price)
  - Time remaining until event
  - Seat quality score
  - Event context (weekend, playoff, day of week)

- **`demand_model_v12_anti_gouge.json`**: Model metadata and configuration (feature names, model parameters, etc.)

## Model Details

The demand model is a logistic regression (binomial GLM) trained on 50 Boston Celtics games. It uses:
- 21 engineered features
- L2 regularization
- Platt scaling for probability calibration
- Time-weighted sample weights to correct for exposure bias

The model is loaded by the RL environment to compute realistic sale probabilities during training and evaluation.

