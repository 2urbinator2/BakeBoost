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
from logging import INFO
from flwr.common.logger import log


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
        Initialize bakery client.

        Args:
            cid: Client ID (bakery identifier)
        """
        self.cid = cid
        log(INFO, f"Bakery client {cid} initialized")

        # TODO: Replace with real XGBoost model weights
        # For now: dummy weights for testing federated setup
        # Use hash to ensure seed is within numpy's valid range (0 to 2^32-1)
        seed = hash(cid) % (2**32)
        np.random.seed(seed)
        self.weights = [
            np.random.rand(10, 5),  # Feature weights
            np.random.rand(5)       # Bias terms
        ]

    def fit(self, parameters, config):
        """
        Train model on bakery's local sales data.

        In production, this would:
        1. Load bakery's historical sales data
        2. Train XGBoost forecasting model
        3. Return updated model parameters

        Args:
            parameters: Global model parameters from server
            config: Training configuration (epochs, batch size, etc.)

        Returns:
            Tuple of (updated_weights, num_training_examples, metrics)
        """
        log(INFO, f"Bakery {self.cid}: Training on local sales data...")

        # TODO: Implement actual training logic:
        # 1. Load data from data/bakery_{cid}_sales.csv
        # 2. Preprocess time series (windowing, normalization)
        # 3. Train XGBoost model
        # 4. Return trained model weights

        # For now: Simulate training by adding small random updates
        updated_weights = [
            w + np.random.rand(*w.shape) * 0.1
            for w in self.weights
        ]

        num_examples = 100  # TODO: Replace with actual training data size
        metrics = {}  # TODO: Add training loss/accuracy

        log(INFO, f"Bakery {self.cid}: Training completed")
        return updated_weights, num_examples, metrics

    def evaluate(self, parameters, config):
        """
        Evaluate model on bakery's local test data.

        In production, this would:
        1. Load bakery's test data
        2. Make predictions with current model
        3. Calculate metrics (MAE, RMSE, MAPE)

        Args:
            parameters: Current global model parameters
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_test_examples, metrics_dict)
        """
        log(INFO, f"Bakery {self.cid}: Evaluating model on local test data...")

        # TODO: Implement actual evaluation:
        # 1. Load test data
        # 2. Generate predictions
        # 3. Calculate MAE, RMSE, MAPE

        # For now: Return dummy metrics
        loss = float(np.random.rand())  # TODO: Real evaluation loss
        num_examples = 50  # TODO: Actual test set size
        metrics = {
            "mae": 10.5,  # Mean Absolute Error (in sales units)
            "rmse": 15.2,  # Root Mean Squared Error
            "mape": 8.3    # Mean Absolute Percentage Error
        }

        log(INFO, f"Bakery {self.cid}: Evaluation completed (Loss: {loss:.4f})")
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
