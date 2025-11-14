"""
Glean Federated Client Application

This module implements the federated learning client for individual bakeries.
Each client trains a demand forecasting model locally on its own sales data
and only shares model updates (not raw data) with the federated server.

Privacy-preserving by design: Raw sales data never leaves the bakery.
"""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import numpy as np
from logging import INFO, WARNING
from flwr.common.logger import log

# Import Glean modules for real XGBoost training
from glean.data_loader import load_store_data
from glean.task import train_model, evaluate_model
from glean.utils import model_to_parameters, parameters_to_model


class BakeryClient(NumPyClient):
    """
    Federated learning client representing a single bakery.

    Each bakery:
    1. Trains a forecasting model on its local sales history
    2. Evaluates the model on its local test data
    3. Shares only model parameters (not data) with the server
    4. Receives improved global model updates from other bakeries
    """

    def __init__(self, cid: str):
        """
        Initialize bakery client and load store-specific data.

        Args:
            cid: Client ID (Flower assigns random large IDs)
        """
        self.cid = cid

        # Map Flower's random client ID to one of our 9 stores (0-8)
        # Use hash to ensure consistent mapping
        store_num = hash(cid) % 9
        self.store_id = f"store_{store_num}"

        log(INFO, f"Initializing client {cid} → {self.store_id}")

        # Load data for THIS store only
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = load_store_data(self.store_id)
            self.has_data = True
            log(INFO, f"{self.store_id}: Data loaded successfully")
        except Exception as e:
            log(WARNING, f"{self.store_id}: Failed to load data - {e}")
            self.has_data = False
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None

    def fit(self, parameters, config):
        """
        Train XGBoost model on store's local sales data.

        This implements real federated learning:
        1. Trains XGBoost on THIS store's data only
        2. Serializes trained model to parameters
        3. Returns parameters to server for aggregation

        Args:
            parameters: Global model parameters from server (currently unused)
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_training_examples, metrics)
        """
        # If no data available, return empty
        if not self.has_data:
            log(WARNING, f"{self.store_id}: No data - skipping training")
            return [], 0, {}

        log(INFO, f"{self.store_id}: Training XGBoost on {len(self.X_train)} samples...")

        # Train XGBoost model on local data
        model = train_model(self.X_train, self.y_train)

        # Serialize model to parameters for federated aggregation
        updated_parameters = model_to_parameters(model)

        # Return actual training set size
        num_examples = len(self.X_train)

        log(INFO, f"{self.store_id}: Training completed ({num_examples} samples)")

        return updated_parameters, num_examples, {}

    def evaluate(self, parameters, config):
        """
        Evaluate global model on store's local test data.

        This implements real federated evaluation:
        1. Deserializes global model from parameters
        2. Makes predictions on THIS store's test data
        3. Calculates real metrics (MAE, RMSE, MAPE)

        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_test_examples, metrics_dict)
        """
        # If no data available, return empty
        if not self.has_data:
            log(WARNING, f"{self.store_id}: No data - skipping evaluation")
            return 0.0, 0, {}

        log(INFO, f"{self.store_id}: Evaluating on {len(self.X_test)} test samples...")

        # Deserialize global model from parameters
        model = parameters_to_model(parameters)

        if model is None:
            log(WARNING, f"{self.store_id}: Failed to deserialize model - using default metrics")
            return 0.0, 0, {}

        # Calculate real metrics on local test data
        metrics = evaluate_model(model, self.X_test, self.y_test)

        # Use MAE as primary loss metric
        loss = metrics["mae"]
        num_examples = len(self.X_test)

        log(INFO,
            f"{self.store_id}: Evaluation complete - "
            f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%"
        )

        return loss, num_examples, metrics


def client_fn(context: Context):
    """
    Factory function to create a client instance.

    This is called by Flower for each bakery that joins the federation.

    Args:
        context: Flower context containing node configuration

    Returns:
        Configured FlowerClient instance
    """
    bakery_id = str(context.node_id)
    log(INFO, f"Creating client for Bakery {bakery_id}")
    return BakeryClient(cid=bakery_id).to_client()


# Create ClientApp
app = ClientApp(client_fn=client_fn)
