"""
Data Loading and Preprocessing for Glean Federated Learning

Loads bakery sales data from train.csv and prepares features for XGBoost training.
Each store gets its own train/test split for federated learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split
from logging import INFO, WARNING
from flwr.common.logger import log

from prophet import Prophet
from prophet.plot import plot_plotly, add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from math import sqrt

# Set plotting style
available_styles = plt.style.available
if 'seaborn-v0_8-darkgrid' in available_styles:
    plt.style.use('seaborn-v0_8-darkgrid')
elif 'ggplot' in available_styles:
    plt.style.use('ggplot')
elif 'fivethirtyeight' in available_styles:
    plt.style.use('fivethirtyeight')
else:
    log(INFO, 'Using default matplotlib style')

try:
    sns.set_palette("husl")
except:
    pass


def get_data_path():
    """Get absolute path to data directory."""
    return Path(__file__).parent.parent / "data"


def load_all_data(verbose=True):
    """
    Load train, test, and sample submission data with comprehensive analysis.

    This function loads all datasets and provides detailed information about:
    - Dataset shapes and columns
    - Data types
    - Missing values
    - Summary statistics

    Args:
        verbose: If True, print detailed information about the datasets

    Returns:
        Tuple of (train_data, test_data, sample_submission)
    """
    data_path = get_data_path()

    # Load datasets
    train_data = pd.read_csv(data_path / 'train.csv')
    test_data = pd.read_csv(data_path / 'test.csv')
    sample_submission = pd.read_csv(data_path / 'sample_submission.csv')

    if verbose:
        log(INFO, f'Train shape: {train_data.shape}')
        log(INFO, f'Test shape: {test_data.shape}')
        log(INFO, f'Submission shape: {sample_submission.shape}')
        log(INFO, f'\nTrain columns: {train_data.columns.tolist()}')

        log(INFO, '\nFirst few rows of training data:')
        log(INFO, f'\n{train_data.head()}')

        log(INFO, '\nTraining data summary statistics:')
        log(INFO, f'\n{train_data.describe()}')

        log(INFO, '\nData types:')
        log(INFO, f'\n{train_data.dtypes}')

        log(INFO, '\nMissing values in train:')
        missing_vals = train_data.isnull().sum()
        if missing_vals.sum() > 0:
            log(INFO, f'\n{missing_vals[missing_vals > 0]}')
        else:
            log(INFO, 'No missing values found!')

    return train_data, test_data, sample_submission


def get_num_stores():
    """
    Calculate number of unique stores in training data.

    Returns:
        int: Number of unique stores
    """
    data_path = get_data_path() / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)
    num_stores = df["store"].nunique()

    log(INFO, f"Found {num_stores} unique stores in training data")
    return num_stores


def load_store_data(store_id: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load and prepare data for a specific store.

    Args:
        store_id: Store identifier (e.g., "store_0", "store_1")
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Raises:
        FileNotFoundError: If train.csv doesn't exist
        ValueError: If store has no data or insufficient samples
    """
    data_path = get_data_path() / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    # Load full dataset
    df = pd.read_csv(data_path)

    # Filter for this store
    df_store = df[df["store"] == store_id].copy()

    if len(df_store) == 0:
        raise ValueError(f"No data found for {store_id}")

    if len(df_store) < 10:
        log(WARNING, f"{store_id}: Only {len(df_store)} samples - model may underperform")

    log(INFO, f"{store_id}: Loaded {len(df_store)} samples")

    # Prepare features
    X, y = prepare_features(df_store)

    # Train/test split (chronological order preserved via shuffle=False for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False  # Preserve time series order
    )

    log(INFO, f"{store_id}: Train size={len(X_train)}, Test size={len(X_test)}")

    return X_train, X_test, y_train, y_test


def prepare_features(df: pd.DataFrame):
    """
    Engineer features from raw data for XGBoost training.

    Features created:
    - Date-derived: day_of_week, month, day_of_month, is_weekend
    - Holiday flags: is_state_holiday, is_school_holiday, is_special_day
    - Weather: temperature_max, temperature_min, temperature_mean,
               sunshine_sum, precipitation_sum

    Args:
        df: DataFrame with raw store data

    Returns:
        Tuple of (X, y) where X is feature matrix, y is target (sales)
    """
    df = df.copy()

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Extract date features
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Holiday features - convert text to binary (0/1)
    # Values are like "normal_day", "state_holiday", "school_holiday", "special_day"
    df["is_state_holiday"] = (df["is_state_holiday"] != "normal_day").astype(int)
    df["is_school_holiday"] = (df["is_school_holiday"] != "normal_day").astype(int)
    df["is_special_day"] = (df["is_special_day"] != "normal_day").astype(int)

    # Weather features - handle missing values
    weather_cols = [
        "temperature_max", "temperature_min", "temperature_mean",
        "sunshine_sum", "precipitation_sum"
    ]

    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            # If weather column missing, add with default value
            df[col] = 0.0

    # Select features for model
    feature_cols = [
        "day_of_week", "month", "day_of_month", "is_weekend",
        "is_state_holiday", "is_school_holiday", "is_special_day",
        "temperature_max", "temperature_min", "temperature_mean",
        "sunshine_sum", "precipitation_sum"
    ]

    X = df[feature_cols].copy()

    # Target variable
    if "sales" not in df.columns:
        raise ValueError("Target column 'sales' not found in data")

    y = df["sales"].copy()

    # Handle missing target values
    missing_targets = y.isna().sum()
    if missing_targets > 0:
        log(WARNING, f"Dropping {missing_targets} rows with missing sales values")
        mask = y.notna()
        X = X[mask]
        y = y[mask]

    # Ensure no NaN values remain
    X = X.fillna(0)

    return X, y


def get_store_statistics():
    """
    Get summary statistics for all stores in the dataset.

    Returns:
        DataFrame with columns: store, num_samples, mean_sales, std_sales
    """
    data_path = get_data_path() / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    df = pd.read_csv(data_path)

    stats = df.groupby("store")["sales"].agg([
        ("num_samples", "count"),
        ("mean_sales", "mean"),
        ("std_sales", "std"),
        ("min_sales", "min"),
        ("max_sales", "max")
    ]).reset_index()

    return stats
