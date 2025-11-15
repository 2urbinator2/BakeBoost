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
# from prophet.plot import plot_plotly, add_changepoints_to_plot  # Commented out - plotly not needed
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


def handle_missing_values(df):
    """
    Handle missing values in both numeric and categorical columns.

    - Numeric columns: filled with median
    - Categorical columns: filled with mode (or 'Unknown' if no mode exists)

    Args:
        df: DataFrame to handle missing values for

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    # Handle numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Handle categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)

    return df


def preprocess_dates(df, date_col='date', sort=True, verbose=True):
    """
    Convert date column to datetime and optionally sort by date.

    Args:
        df: DataFrame to preprocess
        date_col: Name of the date column
        sort: If True, sort DataFrame by date
        verbose: If True, print date range information

    Returns:
        DataFrame with processed dates
    """
    df = df.copy()

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort by date
    if sort:
        df = df.sort_values(date_col)

    if verbose:
        log(INFO, f'Date range: {df[date_col].min()} to {df[date_col].max()}')

    return df


def create_time_features(df, date_col='date'):
    """
    Create comprehensive time-based features from date column.

    Creates:
    - Basic time components: year, month, day, dayofweek, week, quarter, dayofyear
    - Cyclical features: month_sin/cos, day_sin/cos, dayofweek_sin/cos
    - Binary flags: is_weekend, is_month_start, is_month_end

    Args:
        df: DataFrame with date column
        date_col: Name of the date column (must be datetime type)

    Returns:
        DataFrame with added time features
    """
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Extract time components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['week'] = df[date_col].dt.isocalendar().week.astype(int)
    df['quarter'] = df[date_col].dt.quarter
    df['dayofyear'] = df[date_col].dt.dayofyear

    # Cyclical features (sine/cosine encoding for circular time features)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Binary flags
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = (df['day'] <= 3).astype(int)
    df['is_month_end'] = (df['day'] >= 28).astype(int)

    return df


def encode_categorical_features(train_df, test_df=None, date_col='date', exclude_cols=None):
    """
    Encode categorical features using LabelEncoder.

    Fits encoders on combined train+test data to ensure consistency.
    Creates new columns with '_encoded' suffix.

    Args:
        train_df: Training DataFrame
        test_df: Optional test DataFrame (if None, only encodes train)
        date_col: Date column to exclude from encoding
        exclude_cols: Additional columns to exclude from encoding

    Returns:
        Tuple of (train_df, test_df, label_encoders) or (train_df, label_encoders) if test_df is None
    """
    train_df = train_df.copy()
    if test_df is not None:
        test_df = test_df.copy()

    # Find categorical columns
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    # Exclude date column and any other specified columns
    exclude_list = [date_col] if date_col in categorical_cols else []
    if exclude_cols:
        exclude_list.extend(exclude_cols)

    categorical_cols = [col for col in categorical_cols if col not in exclude_list]

    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()

        # Fit on combined train+test values to ensure consistency
        if test_df is not None:
            combined_values = pd.concat([train_df[col], test_df[col]]).unique()
        else:
            combined_values = train_df[col].unique()

        le.fit(combined_values)

        # Transform train data
        train_df[col + '_encoded'] = le.transform(train_df[col])

        # Transform test data if provided
        if test_df is not None:
            test_df[col + '_encoded'] = le.transform(test_df[col])

        label_encoders[col] = le

    log(INFO, f'Encoded {len(categorical_cols)} categorical columns: {categorical_cols}')

    if test_df is not None:
        return train_df, test_df, label_encoders
    else:
        return train_df, label_encoders


def load_all_data(verbose=True, preprocess=True, create_features=True, encode_categorical=True):
    """
    Load train, test, and sample submission data with comprehensive analysis.

    This function loads all datasets and provides detailed information about:
    - Dataset shapes and columns
    - Data types
    - Missing values
    - Summary statistics

    Optionally preprocesses the data by:
    - Converting dates to datetime format
    - Sorting by date
    - Handling missing values
    - Creating time-based features
    - Encoding categorical variables

    Args:
        verbose: If True, print detailed information about the datasets
        preprocess: If True, apply basic preprocessing steps
        create_features: If True, create time-based features
        encode_categorical: If True, encode categorical features

    Returns:
        Tuple of (train_data, test_data, sample_submission, label_encoders)
        If encode_categorical=False, label_encoders will be None
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

    label_encoders = None

    # Apply preprocessing if requested
    if preprocess:
        # Preprocess dates
        if verbose:
            log(INFO, '\nPreprocessing train data dates...')
        train_data = preprocess_dates(train_data, date_col='date', verbose=verbose)

        if verbose:
            log(INFO, '\nPreprocessing test data dates...')
        test_data = preprocess_dates(test_data, date_col='date', verbose=verbose)

        # Handle missing values
        if verbose:
            log(INFO, '\nHandling missing values...')
        train_data = handle_missing_values(train_data)
        test_data = handle_missing_values(test_data)

        if verbose:
            log(INFO, 'Missing values handled successfully!')

    # Create time features if requested
    if create_features:
        if verbose:
            log(INFO, '\nCreating time features...')
        train_data = create_time_features(train_data, date_col='date')
        test_data = create_time_features(test_data, date_col='date')

        if verbose:
            log(INFO, f'Time features created. New column count: {len(train_data.columns)}')

    # Encode categorical features if requested
    if encode_categorical:
        if verbose:
            log(INFO, '\nEncoding categorical features...')
        # encode_categorical_features returns (train, test, encoders) when test_df is provided
        result = encode_categorical_features(train_data, test_data, date_col='date')
        train_data, test_data, label_encoders = result  # type: ignore

    return train_data, test_data, sample_submission, label_encoders


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


def load_store_data(store_id: str, test_size: float = 0.2, random_state: int = 42,
                    create_features: bool = True, encode_categorical: bool = True):
    """
    Load and prepare data for a specific store with comprehensive feature engineering.

    IMPORTANT FOR FEDERATED LEARNING:
    - Categorical encoders are fitted on ALL training data (all stores combined)
    - This ensures ALL stores have the same feature set and model architecture
    - Critical for federated aggregation to work correctly

    Pipeline:
    1. Load ALL training data and preprocess
    2. Fit encoders on ALL data to ensure consistency across stores
    3. Filter for specific store
    4. Extract features and target
    5. Split into train/test sets

    Args:
        store_id: Store identifier (e.g., "store_0", "store_1")
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        create_features: If True, create advanced time features
        encode_categorical: If True, encode categorical variables

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)

    Raises:
        FileNotFoundError: If train.csv doesn't exist
        ValueError: If store has no data or insufficient samples
    """
    data_path = get_data_path() / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    # Load full dataset (ALL stores - critical for consistent encoding)
    df = pd.read_csv(data_path)

    # Preprocess dates and handle missing values on ALL data
    df = preprocess_dates(df, date_col='date', sort=True, verbose=False)
    df = handle_missing_values(df)

    # Create time features on ALL data if requested
    if create_features:
        df = create_time_features(df, date_col='date')

    # Encode categorical features on ALL data to ensure consistency across stores
    # This is CRITICAL for federated learning - all stores must have same features
    if encode_categorical:
        result = encode_categorical_features(df, test_df=None, date_col='date')
        df, _ = result  # Returns (df, encoders) when test_df is None

    # NOW filter for this specific store (after encoding on all data)
    df_store = df[df["store"] == store_id].copy()

    if len(df_store) == 0:
        raise ValueError(f"No data found for {store_id}")

    if len(df_store) < 10:
        log(WARNING, f"{store_id}: Only {len(df_store)} samples - model may underperform")

    log(INFO, f"{store_id}: Loaded {len(df_store)} samples")

    # Extract features and target
    X, y = extract_features_and_target(df_store, target_col='sales')

    # Train/test split (chronological order preserved via shuffle=False for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=False  # Preserve time series order
    )

    log(INFO, f"{store_id}: Train size={len(X_train)}, Test size={len(X_test)}")

    return X_train, X_test, y_train, y_test


def extract_features_and_target(df: pd.DataFrame, target_col='sales', exclude_cols=None):
    """
    Extract feature matrix (X) and target variable (y) from preprocessed DataFrame.

    Assumes that feature engineering has already been done using:
    - create_time_features()
    - encode_categorical_features()
    - handle_missing_values()

    Args:
        df: DataFrame with engineered features
        target_col: Name of target column (default: 'sales')
        exclude_cols: List of columns to exclude from features (e.g., ['date', 'store', 'id'])

    Returns:
        Tuple of (X, y) where X is feature matrix, y is target variable
    """
    df = df.copy()

    # Target variable
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    y = df[target_col].copy()

    # Default columns to exclude
    if exclude_cols is None:
        exclude_cols = []

    # Always exclude target, date, and identifier columns
    default_excludes = [target_col, 'date', 'store', 'id']
    exclude_cols = list(set(exclude_cols + default_excludes))

    # Also exclude original categorical columns (keep only encoded versions)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(categorical_cols)

    # Select feature columns (everything except excluded)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()

    # Handle missing target values
    missing_targets = y.isna().sum()
    if missing_targets > 0:
        log(WARNING, f"Dropping {missing_targets} rows with missing target values")
        mask = y.notna()
        X = X[mask]
        y = y[mask]

    # Ensure no NaN values remain in features
    X = X.fillna(0)

    log(INFO, f"Extracted {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    return X, y


def select_numeric_features(train_data, test_data, target_col='sales', date_col='date',
                            exclude_cols=None):
    """
    Select common numeric features between train and test datasets for XGBoost.

    Automatically excludes:
    - Target column
    - Date column
    - Store identifier
    - Original categorical columns (keeps encoded versions)

    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        target_col: Name of target column to exclude
        date_col: Name of date column to exclude
        exclude_cols: Additional columns to exclude

    Returns:
        List of selected feature column names
    """
    # Find common columns between train and test
    common_cols = set(train_data.columns) & set(test_data.columns)

    # Default exclusions
    default_exclude = [target_col, date_col, 'store', 'id']
    if exclude_cols:
        default_exclude.extend(exclude_cols)

    # Select only numeric columns from common columns
    feature_cols = [
        col for col in common_cols
        if col not in default_exclude
        and train_data[col].dtype in ['int64', 'float64', 'int32', 'float32']
    ]

    log(INFO, f"Selected {len(feature_cols)} numeric features for XGBoost")
    log(INFO, f"Features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    return feature_cols


def create_time_based_split(train_data, date_col='date', split_ratio=0.8):
    """
    Create time-based train/validation split.

    Uses a quantile-based split to preserve chronological order.

    Args:
        train_data: Training DataFrame with date column
        date_col: Name of the date column
        split_ratio: Fraction of data to use for training (e.g., 0.8 = 80% train, 20% val)

    Returns:
        Tuple of (train_idx, val_idx) boolean masks
    """
    split_date = train_data[date_col].quantile(split_ratio)

    train_idx = train_data[date_col] < split_date
    val_idx = train_data[date_col] >= split_date

    log(INFO, f"Time-based split at date: {split_date}")
    log(INFO, f"Training set size: {train_idx.sum()}")
    log(INFO, f"Validation set size: {val_idx.sum()}")

    return train_idx, val_idx


def calculate_rmspe(y_true, y_pred):
    """
    Calculate Root Mean Squared Percentage Error.

    RMSPE is useful for sales forecasting as it measures relative errors.
    Excludes samples where y_true is zero to avoid division by zero.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSPE score (lower is better)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        log(WARNING, "All true values are zero, cannot calculate RMSPE")
        return np.nan

    rmspe = np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))
    return rmspe


def evaluate_predictions(y_true, y_pred, verbose=True):
    """
    Evaluate predictions using multiple metrics.

    Calculates:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - RMSPE (Root Mean Squared Percentage Error)

    Args:
        y_true: True values
        y_pred: Predicted values
        verbose: If True, print metrics

    Returns:
        Dict with metrics: {'rmse': float, 'mae': float, 'rmspe': float}
    """
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    rmspe = calculate_rmspe(y_true, y_pred)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'rmspe': rmspe
    }

    if verbose:
        log(INFO, f"Validation RMSE: {rmse:.4f}")
        log(INFO, f"Validation MAE: {mae:.4f}")
        if not np.isnan(rmspe):
            log(INFO, f"Validation RMSPE: {rmspe:.4f}")

    return metrics


def train_xgboost_model(X_train, y_train, X_val=None, y_val=None,
                        param_grid=None, cv_splits=3, verbose=True):
    """
    Train XGBoost model with grid search and cross-validation.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Optional validation features (for evaluation)
        y_val: Optional validation target (for evaluation)
        param_grid: Dictionary of hyperparameters to search (uses default if None)
        cv_splits: Number of time series cross-validation splits
        verbose: If True, print training progress

    Returns:
        Tuple of (best_model, best_params, metrics_dict)
    """
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

    # Initialize XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    if verbose:
        log(INFO, "Performing grid search for XGBoost...")
        log(INFO, f"Parameter grid: {param_grid}")

    # Grid search
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1 if verbose else 0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = -grid_search.best_score_

    if verbose:
        log(INFO, f"Best parameters: {best_params}")
        log(INFO, f"Best CV RMSE: {sqrt(best_cv_score):.4f}")

    # Evaluate on validation set if provided
    metrics = {'cv_rmse': sqrt(best_cv_score)}

    if X_val is not None and y_val is not None:
        y_pred_val = best_model.predict(X_val)
        val_metrics = evaluate_predictions(y_val, y_pred_val, verbose=verbose)
        metrics.update(val_metrics)

    return best_model, best_params, metrics


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

    stats = df.groupby("store")["sales"].agg(
        num_samples="count",
        mean_sales="mean",
        std_sales="std",
        min_sales="min",
        max_sales="max"
    ).reset_index()

    return stats
