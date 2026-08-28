import numpy as np
import pytest

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


class _FakeWebsocket:
    def __init__(self, response):
        self.response = response
        self.sent = []

    def send(self, data):
        self.sent.append(msgpack_numpy.unpackb(data))

    def recv(self):
        return self.response


def _make_client(response):
    client = object.__new__(websocket_client_policy.WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()
    client._ws = _FakeWebsocket(response)
    return client


def test_infer_batch_sends_explicit_batch_payload():
    response = msgpack_numpy.packb({"actions": np.zeros((2, 3, 1), dtype=np.float32)})
    client = _make_client(response)

    result = client.infer_batch([{"state": np.asarray([1])}, {"state": np.asarray([2])}])

    assert result["actions"].shape == (2, 3, 1)
    assert len(client._ws.sent) == 1
    assert set(client._ws.sent[0]) == {"batch"}
    np.testing.assert_array_equal(client._ws.sent[0]["batch"][0]["state"], np.asarray([1]))
    np.testing.assert_array_equal(client._ws.sent[0]["batch"][1]["state"], np.asarray([2]))


def test_infer_batch_reuses_error_response_handling():
    client = _make_client("traceback")

    with pytest.raises(RuntimeError, match="Error in inference server"):
        client.infer_batch([{"state": np.asarray([1])}])


def test_infer_batch_validates_input():
    response = msgpack_numpy.packb({})
    client = _make_client(response)

    with pytest.raises(ValueError, match="at least one observation"):
        client.infer_batch([])
    with pytest.raises(TypeError, match="sequence of observation dictionaries"):
        client.infer_batch({"state": np.asarray([1])})
    with pytest.raises(TypeError, match="only observation dictionaries"):
        client.infer_batch([{"state": np.asarray([1])}, "bad"])
