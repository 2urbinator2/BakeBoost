"""
Glean Federated Server Application

This module implements the federated learning server that coordinates
training across multiple bakery clients without accessing their raw data.

The server uses Federated Averaging (FedAvg) to aggregate model updates
from each bakery and distribute the improved global model back to clients.
"""

from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Context
from logging import INFO
from flwr.common.logger import log


def server_fn(context: Context) -> ServerAppComponents:
    """
    Create and configure the federated server components.

    Args:
        context: Server context from Flower

    Returns:
        ServerAppComponents with strategy and configuration
    """
    log(INFO, "Glean federated server initialized with FedAvg strategy")
    log(INFO, "Waiting for bakery clients to connect...")

    # Define federated averaging strategy
    strategy = FedAvg(
        fraction_fit=1.0,  # Use 100% of available clients for training each round
        fraction_evaluate=1.0,  # Use 100% of clients for evaluation
        min_fit_clients=3,  # Minimum 3 bakeries needed to start training
        min_evaluate_clients=3,  # Minimum 3 bakeries needed for evaluation
        min_available_clients=3,  # Wait for at least 3 bakeries to connect
    )

    # Server configuration
    config = ServerConfig(num_rounds=20)

    return ServerAppComponents(
        strategy=strategy,
        config=config,
    )


# Create ServerApp with server_fn (new API - avoids deprecation warning)
app = ServerApp(server_fn=server_fn)
