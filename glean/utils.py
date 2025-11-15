"""
Utility Functions for Glean Federated Learning

Handles serialization/deserialization of XGBoost models for Flower's
parameter exchange format (numpy arrays).
"""

import io
import numpy as np
import xgboost as xgb
from typing import List, Optional
from logging import INFO, WARNING
from flwr.common.logger import log


def model_to_parameters(model: xgb.XGBRegressor) -> List[np.ndarray]:
    """
    Serialize XGBoost model to Flower's parameter format.

    Converts XGBoost model to JSON string, then encodes as numpy array
    for transmission in federated learning.

    Args:
        model: Trained XGBoost model

    Returns:
        List containing single numpy array with serialized model
    """
    if model is None:
        log(WARNING, "Attempting to serialize None model")
        return [np.array([])]

    try:
        # Get XGBoost model as JSON string
        model_json = model.get_booster().save_raw(raw_format='json')

        # Convert JSON bytes to numpy array
        model_data = np.frombuffer(model_json, dtype=np.uint8)

        # Wrap in list (Flower expects list of arrays)
        return [model_data]

    except Exception as e:
        log(WARNING, f"Error serializing model: {e}")
        return [np.array([])]


def parameters_to_model(parameters: List[np.ndarray]) -> Optional[xgb.XGBRegressor]:
    """
    Deserialize Flower parameters to XGBoost model.

    Converts numpy array back to XGBoost model by decoding bytes.

    Args:
        parameters: List of numpy arrays from Flower

    Returns:
        XGBoost model, or None if deserialization fails
    """
    if not parameters or len(parameters) == 0:
        log(WARNING, "Received empty parameters")
        return None

    try:
        # Extract numpy array
        model_data = parameters[0]

        if len(model_data) == 0:
            log(WARNING, "Received empty model data")
            return None

        # Convert numpy array back to bytes
        model_json = model_data.tobytes()

        # Create new XGBoost model
        model = xgb.XGBRegressor()

        # Load the booster from serialized data
        # Note: We need to create a booster first, then load into it
        booster = xgb.Booster()
        booster.load_model(bytearray(model_json))

        # Set the booster to the XGBRegressor
        model._Booster = booster

        return model

    except Exception as e:
        log(WARNING, f"Error deserializing model: {e}")
        return None


def get_model_size(parameters: List[np.ndarray]) -> int:
    """
    Calculate size of serialized model in bytes.

    Useful for monitoring network transfer overhead.

    Args:
        parameters: List of numpy arrays

    Returns:
        Total size in bytes
    """
    if not parameters:
        return 0

    total_bytes = sum(arr.nbytes for arr in parameters)
    return total_bytes


def validate_parameters(parameters: List[np.ndarray]) -> bool:
    """
    Validate that parameters are in correct format.

    Args:
        parameters: List of numpy arrays from Flower

    Returns:
        True if valid, False otherwise
    """
    if not isinstance(parameters, list):
        return False

    if len(parameters) == 0:
        return False

    if not isinstance(parameters[0], np.ndarray):
        return False

    if len(parameters[0]) == 0:
        return False

    return True
