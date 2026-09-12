import contextlib
import socket
import threading
import time

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import pytest
import websockets.exceptions
import websockets.sync.client as _sync_client

from openpi.serving import websocket_policy_server


class _EchoPolicy(_base_policy.BasePolicy):
    def infer(self, obs: dict) -> dict:
        del obs
        return {"actions": [0.0]}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_listening(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError as e:
            last_error = e
            time.sleep(0.02)
    raise TimeoutError(f"Server did not start listening on {host}:{port}") from last_error


@contextlib.contextmanager
def _running_server(**server_kwargs):
    port = _free_port()
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=_EchoPolicy(), host="127.0.0.1", port=port, **server_kwargs
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _wait_until_listening("127.0.0.1", port)
    yield port


def _packed_obs(payload_bytes: int) -> bytes:
    packer = msgpack_numpy.Packer()
    return packer.pack({"image": np.zeros(payload_bytes, dtype=np.uint8)})


def test_default_max_size_is_bounded():
    # A `None` max_size disables the websockets library's own message-size
    # protection, letting any connected client force the server to buffer an
    # arbitrarily large message in memory (resource-exhaustion DoS).
    server = websocket_policy_server.WebsocketPolicyServer(policy=_EchoPolicy(), port=0)
    assert server._max_size is not None  # noqa: SLF001
    assert server._max_size > 0  # noqa: SLF001


def test_default_max_size_allows_typical_observation():
    # A real observation (e.g. four aloha-style 224x224x3 uint8 camera frames)
    # must still fit comfortably under the default cap.
    typical_obs_bytes = 4 * 224 * 224 * 3
    server = websocket_policy_server.WebsocketPolicyServer(policy=_EchoPolicy(), port=0)
    assert server._max_size > typical_obs_bytes  # noqa: SLF001


def test_oversized_message_is_rejected():
    with (
        _running_server(max_size=1024) as port,
        _sync_client.connect(f"ws://127.0.0.1:{port}", max_size=None) as client,
    ):
        client.recv()  # initial metadata frame
        client.send(_packed_obs(4096))
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            client.recv()


def test_message_within_limit_is_accepted():
    with (
        _running_server(max_size=1_000_000) as port,
        _sync_client.connect(f"ws://127.0.0.1:{port}") as client,
    ):
        client.recv()  # initial metadata frame
        client.send(_packed_obs(1_000))
        response = client.recv()
        assert response is not None
