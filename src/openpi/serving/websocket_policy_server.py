import asyncio
import http
import logging
import os
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

# How many elements to print when sampling large arrays. Configure via env OPENPI_SAMPLE_MAX.
SAMPLE_MAX = int(os.environ.get("OPENPI_SAMPLE_MAX", "10"))


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                # Log received observation keys and a small sample for debugging.
                try:
                    logger.info("Received obs from %s keys=%s", websocket.remote_address, list(obs.keys()) if isinstance(obs, dict) else None)
                    # Small sample at DEBUG level to avoid spamming INFO logs for large arrays.
                    try:
                        import numpy as _np

                        if isinstance(obs, dict) and "state" in obs:
                            sample = _np.array(obs.get("state"))
                            logger.debug(
                                "obs state shape=%s sample=%s",
                                getattr(sample, "shape", None),
                                str(sample.ravel()[:SAMPLE_MAX]),
                            )
                    except Exception:
                        logger.debug("Could not sample obs values", exc_info=True)
                except Exception:
                    logger.exception("Failed to log incoming obs")

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # Log action keys and a small sample for debugging.
                try:
                    logger.info("Sending action to %s keys=%s", websocket.remote_address, list(action.keys()) if isinstance(action, dict) else None)
                    try:
                        import numpy as _np

                        if isinstance(action, dict) and "actions" in action:
                            sample = _np.array(action.get("actions"))
                            logger.debug(
                                "action['actions'] shape=%s sample=%s",
                                getattr(sample, "shape", None),
                                str(sample.ravel()[:SAMPLE_MAX]),
                            )
                    except Exception:
                        logger.debug("Could not sample action values", exc_info=True)
                except Exception:
                    logger.exception("Failed to log action output")

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
