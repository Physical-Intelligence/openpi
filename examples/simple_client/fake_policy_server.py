import dataclasses
import logging
import time

import numpy as np
from openpi_client import msgpack_numpy
import tyro
import websockets.exceptions
import websockets.sync.server


@dataclasses.dataclass
class Args:
    """Run a CPU-only fake policy server for testing the websocket client."""

    # Host to bind the websocket server to.
    host: str = "0.0.0.0"
    # Port to bind the websocket server to.
    port: int = 8000
    # Number of actions returned in each chunk.
    action_horizon: int = 10
    # Size of each action vector.
    action_dim: int = 8
    # Optional sleep that simulates model latency.
    delay_ms: float = 0.0


def main(args: Args) -> None:
    packer = msgpack_numpy.Packer()
    metadata = {
        "model": "fake_policy",
        "action_horizon": args.action_horizon,
        "action_dim": args.action_dim,
    }

    def handler(websocket: websockets.sync.server.ServerConnection) -> None:
        logging.info("Connection opened")
        websocket.send(packer.pack(metadata))
        request_index = 0

        while True:
            try:
                obs = msgpack_numpy.unpackb(websocket.recv())
                start_time = time.monotonic()
                if args.delay_ms > 0:
                    time.sleep(args.delay_ms / 1000)

                actions = _make_action_chunk(
                    action_horizon=args.action_horizon,
                    action_dim=args.action_dim,
                    request_index=request_index,
                )
                response = {
                    "actions": actions,
                    "server_timing": {"infer_ms": (time.monotonic() - start_time) * 1000},
                    "policy_timing": {"fake_policy_ms": args.delay_ms},
                    "prompt": obs.get("prompt", ""),
                }
                websocket.send(packer.pack(response))
                request_index += 1
            except websockets.exceptions.ConnectionClosed:
                logging.info("Connection closed")
                break

    logging.info("Serving fake policy on %s:%s", args.host, args.port)
    with websockets.sync.server.serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        server.serve_forever()


def _make_action_chunk(*, action_horizon: int, action_dim: int, request_index: int) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, action_horizon, dtype=np.float32)[:, None]
    offsets = np.linspace(0.0, 0.1, action_dim, dtype=np.float32)[None, :]
    return base + offsets + request_index * 0.01


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
