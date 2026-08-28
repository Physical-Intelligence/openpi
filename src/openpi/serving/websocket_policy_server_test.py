import numpy as np
from openpi_client import base_policy

from openpi.serving import websocket_policy_server


class _FakePolicy(base_policy.BasePolicy):
    def __init__(self):
        self.infer_request = None
        self.infer_batch_request = None

    def infer(self, obs):
        self.infer_request = obs
        return {"mode": "single"}

    def infer_batch(self, obs_batch):
        self.infer_batch_request = obs_batch
        return {"mode": "batch", "actions": np.zeros((len(obs_batch), 2, 1), dtype=np.float32)}


def test_infer_policy_request_routes_explicit_batch():
    policy = _FakePolicy()
    request = {"batch": [{"state": np.asarray([1])}, {"state": np.asarray([2])}]}

    result = websocket_policy_server.infer_policy_request(policy, request)

    assert result["mode"] == "batch"
    assert policy.infer_batch_request == request["batch"]
    assert policy.infer_request is None


def test_infer_policy_request_does_not_treat_batch_key_as_magic_when_other_keys_exist():
    policy = _FakePolicy()
    request = {"batch": np.asarray([1]), "state": np.asarray([2])}

    result = websocket_policy_server.infer_policy_request(policy, request)

    assert result["mode"] == "single"
    assert policy.infer_request is request
    assert policy.infer_batch_request is None
