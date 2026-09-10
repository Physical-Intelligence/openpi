import asyncio
import socket
import threading
import time

from openpi_client import websocket_client_policy
import pytest

from openpi.serving import websocket_policy_server

_SECRET = "SECRET_INTERNAL_DETAIL_DO_NOT_LEAK"


class _FailingPolicy:
    def infer(self, obs: dict) -> dict:
        raise RuntimeError(_SECRET)

    def reset(self) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _wait_until_listening(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"server on port {port} did not start listening")


@pytest.fixture
def serve():
    def _serve(**kwargs) -> int:
        port = _free_port()
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=_FailingPolicy(), host="127.0.0.1", port=port, **kwargs
        )
        thread = threading.Thread(target=lambda: asyncio.run(server.run()), daemon=True)
        thread.start()
        _wait_until_listening(port)
        return port

    return _serve


def test_traceback_not_sent_to_client_by_default(serve):
    port = serve()
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)

    with pytest.raises(RuntimeError) as exc_info:
        client.infer({"observation": 1})

    message = str(exc_info.value)
    assert _SECRET not in message
    assert "Traceback (most recent call last)" not in message


def test_traceback_sent_to_client_when_enabled(serve):
    port = serve(send_tracebacks=True)
    client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=port)

    with pytest.raises(RuntimeError) as exc_info:
        client.infer({"observation": 1})

    message = str(exc_info.value)
    assert _SECRET in message
    assert "Traceback (most recent call last)" in message
