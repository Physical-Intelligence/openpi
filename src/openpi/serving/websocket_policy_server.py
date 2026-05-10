import asyncio
import concurrent.futures
import http
import logging
import time
import traceback
from typing import Any

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


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
        inference_workers: int = 1,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        if inference_workers < 1:
            raise ValueError(f"inference_workers must be >= 1, got {inference_workers}")

        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(
            max_workers=inference_workers,
            thread_name_prefix="openpi-policy-infer",
        )
        self._owns_executor = executor is None
        self._infer_semaphore = asyncio.Semaphore(inference_workers)
        self._closed = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_executor:
            self._executor.shutdown(wait=True, cancel_futures=True)

    async def _infer_async(self, obs: dict[str, Any]) -> dict[str, Any]:
        await self._infer_semaphore.acquire()
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, self._policy.infer, obs)
        except Exception:
            self._infer_semaphore.release()
            raise

        future.add_done_callback(self._release_infer_slot)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            future.add_done_callback(self._log_late_infer_error)
            raise

    def _release_infer_slot(self, _future: asyncio.Future) -> None:
        self._infer_semaphore.release()

    def _log_late_infer_error(self, future: asyncio.Future) -> None:
        if future.cancelled():
            return
        try:
            future.result()
        except Exception as exc:
            logger.warning(
                "Policy inference failed after the websocket request was cancelled.",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                action = await self._infer_async(obs)
                infer_time = time.monotonic() - infer_time

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
