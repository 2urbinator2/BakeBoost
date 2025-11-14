"""
XGBoost Model Training and Evaluation for Glean

Defines the machine learning model and training/evaluation logic
for bakery demand forecasting in federated learning.
"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def get_model(random_state: int = 42):
    """
    Create and configure XGBoost regressor for demand forecasting.

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Configured XGBRegressor instance
    """
    model = xgb.XGBRegressor(
        n_estimators=100,        # Number of boosting rounds
        max_depth=5,             # Maximum tree depth
        learning_rate=0.1,       # Step size shrinkage
        objective='reg:squarederror',  # Loss function for regression
        random_state=random_state,
        n_jobs=1,                # Single thread (federated clients run in parallel)
        verbosity=0              # Suppress XGBoost output
    )

    return model


def train_model(X_train, y_train, random_state: int = 42):
    """
    Train XGBoost model on bakery sales data.

    Args:
        X_train: Training features (pandas DataFrame or numpy array)
        y_train: Training target (sales values)
        random_state: Random seed for reproducibility

    Returns:
        Trained XGBRegressor model
    """
    model = get_model(random_state=random_state)

    # Train model
    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate trained model on test data.

    Calculates three key metrics for demand forecasting:
    - MAE (Mean Absolute Error): Average absolute difference (in sales units)
    - RMSE (Root Mean Squared Error): Penalizes large errors more
    - MAPE (Mean Absolute Percentage Error): Percentage error metric

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target (actual sales)

    Returns:
        Dictionary with keys: 'mae', 'rmse', 'mape'
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # MAPE calculation with protection against division by zero
    # Avoid MAPE if y_test has zeros (undefined)
    if (y_test == 0).any():
        # Use epsilon to avoid division by zero
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
    else:
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape)
    }

    return metrics


def make_predictions(model, X):
    """
    Make sales predictions using trained model.

    Args:
        model: Trained XGBoost model
        X: Feature matrix

    Returns:
        numpy array of predicted sales values
    """
    predictions = model.predict(X)
    return predictions
