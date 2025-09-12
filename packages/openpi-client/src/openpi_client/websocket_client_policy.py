import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        """Initialize the websocket client policy.

        Args:
            host: The hostname or IP address of the server to connect to.
            port: The port number to connect to. If None, no port is appended to the URI.
            api_key: Optional API key for authentication. If provided, it will be sent in the Authorization header.
        """
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        """Get metadata received from the server during connection.

        Returns:
            Dictionary containing server metadata that was received during the initial connection.
        """
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Establish connection to the server and retrieve metadata.

        Continuously attempts to connect to the server until successful, with 5-second intervals between attempts.
        Once connected, receives and unpacks the server metadata.

        Returns:
            Tuple containing the websocket connection and the server metadata dictionary.

        Raises:
            Any exception that occurs during metadata unpacking or connection establishment (except ConnectionRefusedError).
        """
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Send observation to server and receive inference result.

        Args:
            obs: Dictionary containing observation data to be sent to the server.

        Returns:
            Dictionary containing the inference result from the server.

        Raises:
            RuntimeError: If the server responds with an error message (string instead of bytes).
        """
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        """Reset the policy state.

        This implementation does nothing as the websocket client maintains no local state that needs resetting.
        """
        pass
