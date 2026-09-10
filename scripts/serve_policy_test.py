import os

os.environ["JAX_PLATFORMS"] = "cpu"

from . import serve_policy


class _FakePolicy:
    def __init__(self) -> None:
        self.metadata: dict = {}

    def infer(self, obs: dict) -> dict:
        return {}


def test_default_host_is_loopback():
    assert serve_policy.Args().host == "127.0.0.1"


def test_tracebacks_not_sent_by_default():
    assert serve_policy.Args().debug_send_tracebacks is False


def test_main_wires_host_and_traceback_flag(monkeypatch):
    captured: dict = {}

    class _FakeServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def serve_forever(self) -> None:
            pass

    monkeypatch.setattr(serve_policy.websocket_policy_server, "WebsocketPolicyServer", _FakeServer)
    monkeypatch.setattr(serve_policy, "create_policy", lambda args: _FakePolicy())

    serve_policy.main(serve_policy.Args(host="0.0.0.0", debug_send_tracebacks=True))

    assert captured["host"] == "0.0.0.0"
    assert captured["send_tracebacks"] is True
