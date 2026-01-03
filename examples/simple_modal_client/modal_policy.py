"""
Client policy to run remote inference in a Modal GPU container.

Deploy the Modal app which will serve the inference. This only need to be run once.
```
uv run --with quic-portal[modal]==0.1.7 modal deploy examples/simple_modal_client/modal_policy.py
```
"""

import time

import numpy as np
from openpi_client import base_policy as _base_policy, msgpack_numpy
import random
from quic_portal import Portal

import modal

APP_NAME = "openpi-policy-server"

app = modal.App(APP_NAME)

local_ignores = ["./.venv/**", "./.git/**", "./examples/**", "**/__pycache__/**"]

openpi_image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "git-lfs", "linux-libc-dev", "build-essential", "clang")
    .add_local_dir(".", "/root/openpi", ignore=local_ignores, copy=True)
    .pip_install("uv")
    .run_commands(
        "cd /root/openpi && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 UV_PROJECT_ENVIRONMENT='/usr/local/' uv sync"
    )
    .run_commands("cd /root && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system -e openpi")
    .run_commands("cd /root && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system modal")
    .run_commands("cd /root && unset UV_INDEX_URL && GIT_LFS_SKIP_SMUDGE=1 uv pip install --system quic-portal==0.1.7")
)

volume = modal.Volume.from_name("openpi-cache", create_if_missing=True)


@app.cls(
    image=openpi_image,
    region="us-west-1",
    timeout=3600,
    scaledown_window=600,
    gpu="h100",
    volumes={"/root/.cache/openpi": volume},
)
class ModalPolicyClass:
    policy_name: str = modal.parameter(default="pi0_aloha_sim")
    policy_checkpoint: str = modal.parameter(default="s3://openpi-assets/checkpoints/pi0_aloha_sim")

    @modal.enter()
    def enter(self):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        t0 = time.time()
        print("[server] Creating policy ...")
        self.policy = _policy_config.create_trained_policy(
            _config.get_config(self.policy_name), self.policy_checkpoint, default_prompt=None
        )
        print(f"[server] Policy created in {time.time() - t0:.2f}s.")

    @modal.method()
    def serve(self, rendezvous: modal.Dict):
        random_port = random.randint(5555, 65535)
        portal = Portal.create_server(rendezvous, local_port=random_port)
        packer = msgpack_numpy.Packer()

        assert portal.recv() == b"hello"
        portal.send(packer.pack(self.policy.metadata))
        print("[server] Server metadata sent.")

        while True:
            msg = portal.recv()

            if msg == b"ping":
                continue
            if msg == b"exit":
                break

            obs = msgpack_numpy.unpackb(msg)
            action = self.policy.infer(obs)
            portal.send(packer.pack(action))


class ModalPolicy(_base_policy.BasePolicy):
    def __init__(self, policy_name: str, policy_checkpoint: str):
        try:
            ModelClass = modal.Cls.from_name(APP_NAME, "ModalPolicyClass")
            fn = ModelClass(policy_name=policy_name, policy_checkpoint=policy_checkpoint).serve
        except modal.exception.NotFoundError:
            raise Exception(
                "Please first deploy the app with "
                "`uv run --with quic-portal[modal]==0.1.7 modal deploy examples/simple_modal_client/modal_policy.py`"
            ) from None

        self._packer = msgpack_numpy.Packer()
        for attempt in range(5):
            handle = None
            try:
                with modal.Dict.ephemeral() as rendezvous:
                    handle = fn.spawn(rendezvous)
                    random_port = random.randint(5555, 65535)
                    portal = Portal.create_client(rendezvous, local_port=random_port)

                portal.send(b"hello")
                self._server_metadata = msgpack_numpy.unpackb(portal.recv())
                self._portal = portal
                return
            except Exception as e:
                print(f"[client] Still connecting to server ... ")
                if handle:
                    handle.cancel()
                if attempt == 4:
                    raise e

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: np.ndarray) -> np.ndarray:
        self._portal.send(self._packer.pack(obs))
        return msgpack_numpy.unpackb(self._portal.recv())

    def close(self):
        self._portal.send(b"exit")
