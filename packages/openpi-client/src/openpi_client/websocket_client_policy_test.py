from unittest import mock

import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        ("localhost", 8000, "ws://localhost:8000"),
        ("ws-policy.example.com", 8000, "ws://ws-policy.example.com:8000"),
        ("wss-policy.example.com", None, "ws://wss-policy.example.com"),
        ("wss://policy.example.com/infer?robot=left", 8443, "wss://policy.example.com:8443/infer?robot=left"),
        ("ws://localhost:8000/infer", 9000, "ws://localhost:9000/infer"),
        ("ws://[::1]/infer", 8000, "ws://[::1]:8000/infer"),
        ("[::1]", 8000, "ws://[::1]:8000"),
        ("wss://policy.example.com/infer?robot=left", None, "wss://policy.example.com/infer?robot=left"),
    ],
)
def test_connection_uses_a_valid_websocket_uri(host, port, expected):
    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.packb({"robot": "test"})
    with mock.patch("websockets.sync.client.connect", return_value=connection) as connect:
        policy = websocket_client_policy.WebsocketClientPolicy(host, port)
    connect.assert_called_once_with(expected, compression=None, max_size=None, additional_headers=None)
    assert policy.get_server_metadata() == {"robot": "test"}
