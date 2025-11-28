from abc import ABC
from abc import abstractmethod
from enum import Enum
import logging
from typing import Any

import numpy as np


class TrackingBackend(Enum):
    """Available experiment tracking backends."""

    WANDB = "wandb"
    MLFLOW = "mlflow"
    NONE = "none"


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values.
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics at a specific training step.

        Args:
            metrics: Dictionary of metric names and values.
            step: Training step number.
        """

    @abstractmethod
    def log_images(self, images: list[np.ndarray], name: str, step: int) -> None:
        """Log images at a specific training step.

        Args:
            images: List of images to log.
            name: Name/key for the logged images.
            step: Training step number.
        """

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the experiment run.

        Args:
            status: Run status - "FINISHED", "FAILED", or "KILLED".
        """


class WandBTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(self):
        """Initialize WandB tracker."""
        import wandb

        self._wandb = wandb
        logging.info("[TRACKER] Using Weights & Biases backend")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to wandb config."""
        # WandB logs params during init, so this is a no-op
        # (params are already logged in wandb.init)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to wandb."""
        self._wandb.log(metrics, step=step)

    def log_images(self, images: list[np.ndarray], name: str, step: int) -> None:
        """Log images to wandb."""
        wandb_images = [self._wandb.Image(img) for img in images]
        self._wandb.log({name: wandb_images}, step=step)

    def end_run(self, status: str = "FINISHED") -> None:
        """End wandb run."""
        self._wandb.finish()
        logging.info(f"[TRACKER] Ended WandB run with status: {status}")


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker."""

    def __init__(self, mlflow_client):
        """Initialize MLflow tracker.

        Args:
            mlflow_client: Configured and started MLflow client instance.
        """
        self._client = mlflow_client
        logging.info("[TRACKER] Using MLflow backend")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to MLflow."""
        self._client.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to MLflow."""
        self._client.log_metrics(metrics, step=step)

    def log_images(self, images: list[np.ndarray], name: str, step: int) -> None:
        """Log images to MLflow."""
        for i, img in enumerate(images):
            self._client.log_image(img, f"{name}_{i}.png", step=step)

    def end_run(self, status: str = "FINISHED") -> None:
        """End MLflow run."""
        self._client.end_run(status=status)


class NoOpTracker(ExperimentTracker):
    """No-op tracker that logs nothing (useful for disabling tracking)."""

    def __init__(self):
        """Initialize no-op tracker."""
        logging.info("[TRACKER] Tracking disabled")

    def log_params(self, params: dict[str, Any]) -> None:
        """No-op."""

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """No-op."""

    def log_images(self, images: list[np.ndarray], name: str, step: int) -> None:
        """No-op."""

    def end_run(self, status: str = "FINISHED") -> None:
        """No-op."""


def create_tracker(
    backend: TrackingBackend,
    mlflow_client=None,
) -> ExperimentTracker:
    """Factory function to create an experiment tracker for the specified backend.

    Args:
        backend: The tracking backend to use (WANDB, MLFLOW, or NONE).
        mlflow_client: MLflow client instance (required if backend is MLFLOW).

    Returns:
        An experiment tracker instance for the specified backend.
    """
    if backend == TrackingBackend.WANDB:
        return WandBTracker()
    if backend == TrackingBackend.MLFLOW:
        if mlflow_client is None:
            raise ValueError("mlflow_client must be provided when using MLFLOW backend")
        return MLflowTracker(mlflow_client)
    if backend == TrackingBackend.NONE:
        return NoOpTracker()
    raise ValueError(f"Unknown tracking backend: {backend}")
