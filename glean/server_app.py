"""
Glean Federated Server Application

This module implements the federated learning server that coordinates
training across multiple bakery clients without accessing their raw data.

The server uses a custom XGBoost Ensemble Strategy because XGBoost models
are tree-based and cannot be averaged like neural network weights.
"""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context
from logging import INFO
from flwr.common.logger import log

# Import custom XGBoost strategy
from glean.strategy import XGBoostEnsembleStrategy


def server_fn(context: Context) -> ServerAppComponents:
    """
    Create and configure the federated server components.

    Args:
        context: Server context from Flower

    Returns:
        ServerAppComponents with strategy and configuration
    """
    log(INFO, "Glean federated server initialized with XGBoost Ensemble Strategy")
    log(INFO, "Waiting for bakery clients to connect...")

    # Define XGBoost ensemble strategy
    # With 9 stores, require majority (5) to participate
    strategy = XGBoostEnsembleStrategy(
        fraction_fit=1.0,  # Use 100% of available clients for training each round
        fraction_evaluate=1.0,  # Use 100% of clients for evaluation
        min_fit_clients=5,  # Minimum 5 stores (majority of 9) needed to start training
        min_evaluate_clients=5,  # Minimum 5 stores needed for evaluation
        min_available_clients=5,  # Wait for at least 5 stores to connect
    )

    # Server configuration
    config = ServerConfig(num_rounds=20)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


# Create ServerApp with server_fn (new API - avoids deprecation warning)
app = ServerApp(server_fn=server_fn)
