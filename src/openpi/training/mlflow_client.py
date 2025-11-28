import base64
import dataclasses
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
import etils.epath as epath

BASE64_BLOCK_SIZE = 4


@dataclasses.dataclass
class MLflowConfig:
    """
    Configuration for MLflow tracking client.

    Attributes
    ----------
    tracking_uri : str or None
        MLflow tracking server URI (e.g., "http://mlflow-server:5000").
        If None, uses local mlruns directory.
    experiment_name : str
        Name of the MLflow experiment.
    run_name : str
        Name for the MLflow run (typically model_id or experiment name).
    checkpoint_dir : epath.Path
        Directory where checkpoints and MLflow metadata are stored.
    hyperparameters : dict[str, Any]
        Dictionary of hyperparameters to log at run start.
    tags : dict[str, str]
        Dictionary of tags to set at run start.
    """

    tracking_uri: str | None
    experiment_name: str
    run_name: str
    checkpoint_dir: epath.Path
    hyperparameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    tags: dict[str, str] = dataclasses.field(default_factory=dict)


class MLflowClient:
    """
    MLflow tracking client with automatic Cognito authentication.

    This client handles Cognito authentication for remote MLflow servers,
    automatic token refresh via background daemon thread, and MLflow run
    lifecycle management.

    Parameters
    ----------
    config : MLflowConfig
        MLflow configuration including tracking URI, experiment name, and tags.
    resuming : bool, optional
        Whether to resume from an existing run (default is False).

    Attributes
    ----------
    run_id : str or None
        Current MLflow run ID, or None if not started.
    """

    def __init__(self, config: MLflowConfig, resuming: bool = False):
        """
        Initialize MLflow client.

        Parameters
        ----------
        config : MLflowConfig
            MLflow configuration.
        resuming : bool, optional
            Whether to resume from existing run (default is False).
        """
        self._config = config
        self._resuming = resuming
        self._mlflow = None
        self._run_id = None
        self._token_refresh_thread = None
        self._stop_refresh = threading.Event()

    @property
    def run_id(self) -> str | None:
        """
        Get current MLflow run ID.

        Returns
        -------
        str or None
            Current run ID, or None if not started.
        """
        return self._run_id

    def start(self) -> None:
        """
        Start MLflow run with authentication.

        Authenticates with Cognito (if remote tracking URI is configured),
        starts a new MLflow run or resumes existing one, and logs initial
        hyperparameters and tags. Also starts background token refresh daemon.
        """
        if self._config.tracking_uri:
            token = self._get_mlflow_token_from_env()
            if token:
                logging.info("[MLFLOW] Authentication successful")
                self._start_token_refresh_daemon()
            else:
                logging.warning("[MLFLOW] Authentication failed")

        import mlflow

        self._mlflow = mlflow

        if self._config.tracking_uri:
            logging.info(f"[MLFLOW] Setting tracking URI: {self._config.tracking_uri}")
            self._mlflow.set_tracking_uri(self._config.tracking_uri)

        self._mlflow.set_experiment(self._config.experiment_name)

        mlflow_id_file = self._config.checkpoint_dir / "mlflow_run_id.txt"
        if self._resuming and mlflow_id_file.exists():
            run_id = mlflow_id_file.read_text().strip()
            self._mlflow.start_run(run_id=run_id, log_system_metrics=True)
            logging.info(f"[MLFLOW] Resumed run {run_id}")
        else:
            self._mlflow.start_run(run_name=self._config.run_name, log_system_metrics=True)
            self._config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            mlflow_id_file.write_text(self._mlflow.active_run().info.run_id)
            logging.info(f"[MLFLOW] Started new run: {self._config.run_name}")

        self._run_id = self._mlflow.active_run().info.run_id

        if self._config.hyperparameters:
            self.log_params(self._config.hyperparameters)
        if self._config.tags:
            self.log_tags(self._config.tags)

    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary of parameter names and values to log.
        """
        self._mlflow.log_params(params)

    def log_tags(self, tags: dict[str, str]) -> None:
        """
        Log tags to MLflow.

        Parameters
        ----------
        tags : dict[str, str]
            Dictionary of tag names and values to set.
        """
        for key, value in tags.items():
            self._mlflow.set_tag(key, value)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """
        Log metrics to MLflow.

        Parameters
        ----------
        metrics : dict[str, float]
            Dictionary of metric names and values to log.
        step : int
            Training step number.
        """
        float_metrics = {k: float(v) for k, v in metrics.items()}
        self._mlflow.log_metrics(float_metrics, step=step)

    def log_image(self, image, artifact_file: str, step: int | None = None) -> None:
        """
        Log image to MLflow.

        Parameters
        ----------
        image : numpy.ndarray or PIL.Image
            Image to log.
        artifact_file : str
            Path/name for the artifact (e.g., "camera_view_0.png").
        step : int, optional
            Training step number. If provided, image will be stored under
            "step_{step}/{artifact_file}".
        """
        if step is not None:
            artifact_file = f"step_{step}/{artifact_file}"
        self._mlflow.log_image(image, artifact_file)

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End MLflow run.

        Stops the token refresh daemon and ends the MLflow run with the
        specified status.

        Parameters
        ----------
        status : str, optional
            Run status - "FINISHED", "FAILED", or "KILLED" (default is "FINISHED").
        """
        self._stop_refresh.set()
        if self._token_refresh_thread:
            self._token_refresh_thread.join(timeout=1.0)

        try:
            self._mlflow.end_run(status=status)
            logging.info(f"[MLFLOW] Ended run with status: {status}")
        except Exception as e:
            logging.error(f"[MLFLOW] Failed to end run: {e}")

    def _start_token_refresh_daemon(self) -> None:
        """
        Start background daemon thread to refresh token automatically.

        The daemon checks token expiration every 60 seconds and refreshes
        it when it's within 5 minutes of expiring.
        """
        self._token_refresh_thread = threading.Thread(target=self._token_refresh_loop, daemon=True)
        self._token_refresh_thread.start()
        logging.info("[MLFLOW] Started token refresh daemon")

    def _token_refresh_loop(self) -> None:
        """
        Background loop that refreshes token before expiration.

        Runs every 60 seconds, checking if token is within 5 minutes of
        expiring and refreshing it if needed.
        """
        while not self._stop_refresh.is_set():
            current_token = os.environ.get("MLFLOW_TRACKING_TOKEN")

            if current_token and self._is_token_expired(current_token, buffer_seconds=300):
                logging.info("[MLFLOW] Token expiring soon, refreshing...")
                self._refresh_token()

            self._stop_refresh.wait(timeout=60)

    def _calculate_secret_hash(self, username: str, client_id: str, client_secret: str) -> str:
        """
        Calculate SECRET_HASH for Cognito authentication.

        Parameters
        ----------
        username : str
            Cognito username.
        client_id : str
            Cognito app client ID.
        client_secret : str
            Cognito app client secret.

        Returns
        -------
        str
            Base64-encoded HMAC-SHA256 hash.
        """
        message = username + client_id
        dig = hmac.new(
            client_secret.encode("utf-8"),
            msg=message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        return base64.b64encode(dig).decode()

    def _get_cognito_token(
        self,
        username: str,
        password: str,
        user_pool_id: str,
        client_id: str,
        client_secret: str | None = None,
        region: str = "us-east-1",
    ) -> str | None:
        """
        Get JWT ID token from AWS Cognito.

        Parameters
        ----------
        username : str
            Cognito username.
        password : str
            Cognito password.
        user_pool_id : str
            Cognito user pool ID.
        client_id : str
            Cognito app client ID.
        client_secret : str, optional
            Cognito app client secret.
        region : str, optional
            AWS region (default is "us-east-1").

        Returns
        -------
        str or None
            JWT ID token if successful, None otherwise.
        """
        try:
            client = boto3.client("cognito-idp", region_name=region)

            auth_params = {"USERNAME": username, "PASSWORD": password}

            if client_secret:
                secret_hash = self._calculate_secret_hash(username, client_id, client_secret)
                auth_params["SECRET_HASH"] = secret_hash

            response = client.initiate_auth(
                ClientId=client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters=auth_params,
            )

            id_token = response["AuthenticationResult"]["IdToken"]
            logging.info("[AUTH] Successfully obtained Cognito JWT token")
            return id_token

        except ClientError as e:
            logging.error(f"[AUTH] Cognito authentication failed: {e.response['Error']['Code']}")
            return None

    def _get_mlflow_token_from_env(self) -> str | None:
        """
        Get MLflow authentication token from Cognito.

        Authenticates with AWS Cognito using credentials from environment
        variables and obtains a JWT token for MLflow access.

        Environment variables used:
            MLFLOW_COGNITO_USERNAME: Cognito username
            MLFLOW_COGNITO_PASSWORD: Cognito password
            MLFLOW_COGNITO_USER_POOL_ID: Cognito user pool ID
            MLFLOW_COGNITO_CLIENT_ID: Cognito app client ID
            MLFLOW_COGNITO_CLIENT_SECRET: Cognito app client secret (optional)
            MLFLOW_COGNITO_REGION: AWS region (default: us-east-1)

        Returns
        -------
        str or None
            JWT token if authentication succeeds, None otherwise.
        """
        username = os.environ.get("MLFLOW_COGNITO_USERNAME")
        password = os.environ.get("MLFLOW_COGNITO_PASSWORD")
        user_pool_id = os.environ.get("MLFLOW_COGNITO_USER_POOL_ID")
        client_id = os.environ.get("MLFLOW_COGNITO_CLIENT_ID")
        client_secret = os.environ.get("MLFLOW_COGNITO_CLIENT_SECRET")
        region = os.environ.get("MLFLOW_COGNITO_REGION", "us-east-1")

        if not all([username, password, user_pool_id, client_id]):
            logging.info("[AUTH] MLflow Cognito credentials not found")
            return None

        token = self._get_cognito_token(
            username=username,
            password=password,
            user_pool_id=user_pool_id,
            client_id=client_id,
            client_secret=client_secret,
            region=region,
        )

        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token

        return token

    def _decode_jwt_payload(self, token: str) -> dict | None:
        """
        Decode JWT token payload without verification.

        Parameters
        ----------
        token : str
            JWT token string.

        Returns
        -------
        dict or None
            Decoded payload dict, or None if decoding fails.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            payload = parts[1]
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            return json.loads(decoded)
        except Exception as e:
            logging.error(f"[AUTH] Failed to decode JWT: {e}")
            return None

    def _is_token_expired(self, token: str, buffer_seconds: int = 300) -> bool:
        """
        Check if JWT token is expired or will expire soon.

        Parameters
        ----------
        token : str
            JWT token string.
        buffer_seconds : int, optional
            Consider token expired if it expires within this many seconds
            (default is 300, i.e., 5 minutes).

        Returns
        -------
        bool
            True if token is expired or will expire soon, False otherwise.
        """
        payload = self._decode_jwt_payload(token)
        if not payload or "exp" not in payload:
            return True

        exp_time = payload["exp"]
        current_time = time.time()
        return (exp_time - current_time) <= buffer_seconds

    def _refresh_token(self) -> str | None:
        """
        Refresh MLflow authentication token.

        Obtains a new JWT token from Cognito and updates the
        MLFLOW_TRACKING_TOKEN environment variable.

        Returns
        -------
        str or None
            New JWT token if successful, None otherwise.
        """
        username = os.environ.get("MLFLOW_COGNITO_USERNAME")
        password = os.environ.get("MLFLOW_COGNITO_PASSWORD")
        user_pool_id = os.environ.get("MLFLOW_COGNITO_USER_POOL_ID")
        client_id = os.environ.get("MLFLOW_COGNITO_CLIENT_ID")
        client_secret = os.environ.get("MLFLOW_COGNITO_CLIENT_SECRET")
        region = os.environ.get("MLFLOW_COGNITO_REGION", "us-east-1")

        if not all([username, password, user_pool_id, client_id]):
            logging.error("[AUTH] Cannot refresh token: credentials not found")
            return None

        token = self._get_cognito_token(
            username=username,
            password=password,
            user_pool_id=user_pool_id,
            client_id=client_id,
            client_secret=client_secret,
            region=region,
        )

        if token:
            os.environ["MLFLOW_TRACKING_TOKEN"] = token
            logging.info("[AUTH] Token refreshed successfully")

        return token

    def __enter__(self):
        """
        Context manager entry - start MLflow run.

        Returns
        -------
        MLflowClient
            Self for use in with statement.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - end MLflow run.

        Automatically sets run status to FAILED if an exception occurred,
        otherwise sets status to FINISHED.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Exception traceback if an exception occurred.

        Returns
        -------
        bool
            Always returns False to propagate exceptions.
        """
        if exc_type:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
        return False
