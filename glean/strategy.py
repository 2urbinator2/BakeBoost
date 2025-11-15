"""
Custom Federated Learning Strategy for XGBoost

XGBoost models are tree-based and cannot be averaged like neural network weights.
This strategy uses a simple ensemble approach instead.
"""

from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    EvaluateRes,
    FitIns,
    EvaluateIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from typing import List, Tuple, Optional, Union, Dict
from logging import INFO
from flwr.common.logger import log
import numpy as np


class XGBoostEnsembleStrategy(Strategy):
    """
    Federated strategy for XGBoost that uses ensemble instead of averaging.

    Since XGBoost models are tree-based, we cannot average parameters like
    neural networks. Instead, this strategy:
    1. Collects trained models from clients
    2. Keeps track of the best performing model
    3. Distributes the best model back to clients for evaluation

    This is a simple baseline strategy for federated XGBoost.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ):
        """
        Initialize XGBoost ensemble strategy.

        Args:
            fraction_fit: Fraction of clients used for training
            fraction_evaluate: Fraction of clients used for evaluation
            min_fit_clients: Minimum clients needed for training
            min_evaluate_clients: Minimum clients needed for evaluation
            min_available_clients: Minimum clients that must be available
        """
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        # Track best model across rounds
        self.best_parameters = None
        self.best_loss = float('inf')
        self.round_num = 0

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """
        Initialize global model parameters.

        For XGBoost, we start with no global model and let clients train locally.
        """
        log(INFO, "XGBoost strategy: No initial global parameters (clients train locally)")
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configure clients for training round.

        Args:
            server_round: Current round number
            parameters: Current global model parameters (may be None)
            client_manager: Manager for available clients

        Returns:
            List of (client, fit_instructions) tuples
        """
        self.round_num = server_round
        log(INFO, f"Round {server_round}: Configuring fit for clients")

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create fit instructions (empty parameters - clients train locally)
        fit_ins = FitIns(parameters, {})

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients.

        For XGBoost, we don't average models. Instead, we keep the model
        from the client with the most training samples (indicating more data).

        Args:
            server_round: Current round number
            results: Successful client results
            failures: Failed client results

        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        if not results:
            log(INFO, f"Round {server_round}: No results to aggregate")
            return None, {}

        log(INFO, f"Round {server_round}: Aggregating {len(results)} client models")

        # Find client with most training samples (best model proxy)
        best_result = max(results, key=lambda x: x[1].num_examples)
        best_client, best_fit_res = best_result

        log(INFO,
            f"Round {server_round}: Selected model from client with "
            f"{best_fit_res.num_examples} training samples"
        )

        # Store as global model for evaluation
        self.best_parameters = best_fit_res.parameters

        # Aggregate metrics for monitoring
        total_examples = sum([fit_res.num_examples for _, fit_res in results])

        metrics = {
            "num_clients": len(results),
            "total_examples": total_examples,
            "best_client_examples": best_fit_res.num_examples,
        }

        # Return best model as global model
        return self.best_parameters, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Configure clients for evaluation round.

        Args:
            server_round: Current round number
            parameters: Current global model parameters
            client_manager: Manager for available clients

        Returns:
            List of (client, evaluate_instructions) tuples
        """
        # Don't evaluate if no global model yet
        if parameters is None:
            log(INFO, f"Round {server_round}: Skipping evaluation (no global model)")
            return []

        log(INFO, f"Round {server_round}: Configuring evaluation for clients")

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create evaluate instructions with global model
        evaluate_ins = EvaluateIns(parameters, {})

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation results from clients.

        Args:
            server_round: Current round number
            results: Successful evaluation results
            failures: Failed evaluation results

        Returns:
            Tuple of (aggregated_loss, metrics)
        """
        if not results:
            log(INFO, f"Round {server_round}: No evaluation results")
            return None, {}

        # Weighted average of losses (weighted by test set size)
        total_examples = sum([eval_res.num_examples for _, eval_res in results])

        if total_examples == 0:
            return None, {}

        weighted_loss = sum([
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
        ]) / total_examples

        # Aggregate metrics
        metrics_aggregated = {}

        # Collect all metric keys from first result
        if results and results[0][1].metrics:
            metric_keys = results[0][1].metrics.keys()

            for key in metric_keys:
                # Weighted average for each metric
                weighted_value = sum([
                    eval_res.metrics.get(key, 0) * eval_res.num_examples
                    for _, eval_res in results
                ]) / total_examples

                metrics_aggregated[key] = weighted_value

        log(INFO,
            f"Round {server_round}: Evaluation - "
            f"Loss={weighted_loss:.4f}, "
            f"MAE={metrics_aggregated.get('mae', 0):.2f}, "
            f"RMSE={metrics_aggregated.get('rmse', 0):.2f}, "
            f"MAPE={metrics_aggregated.get('mape', 0):.2f}%"
        )

        return weighted_loss, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """
        Evaluate global model on server side.

        Since we don't have server-side data, we skip this.
        """
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Calculate number of clients for training.

        Returns:
            Tuple of (sample_size, min_num_clients)
        """
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_fit_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """
        Calculate number of clients for evaluation.

        Returns:
            Tuple of (sample_size, min_num_clients)
        """
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_evaluate_clients
