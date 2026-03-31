"""WebSocket server that exposes a policy for remote inference.

Overview
--------
``WebsocketPolicyServer`` accepts one client connection at a time and serves
``infer`` requests over a msgpack-encoded WebSocket protocol.  See
``openpi_client.websocket_client_policy`` for the matching client.

Timing integration
------------------
When the wrapped policy implements the ``TaskLifecycle`` protocol (i.e. it
has ``on_task_begin`` and ``on_task_end`` methods — as ``InferenceInterceptor``
does), the server calls:

* ``on_task_begin()`` when a client connection **opens**.
* ``on_task_end()`` when a client connection **closes** (or errors out).

``on_task_end()`` triggers ``SystemTimer.on_task_end()``, which prints a
per-probe timing summary to the terminal and writes a CSV (if configured).

When the policy does *not* implement ``TaskLifecycle`` (plain ``Policy``
without ``--cache``), the ``hasattr`` checks are simply skipped — no
behaviour change for the non-cache path.

Note on ``stage_timing`` removal
---------------------------------
The Step 1 design collected per-inference ``stage_timing`` dicts from the
action output and aggregated them manually at connection close.  That logic
has been removed in Step 2.  Timing aggregation is now entirely the
responsibility of ``SystemTimer``, which is called via the ``TaskLifecycle``
hooks above.  The ``server_timing`` key (wall-clock infer time measured by
this server) is still present in every action response.

Currently only the ``load`` and ``infer`` methods of the client protocol
are implemented.
"""

import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the WebSocket protocol.

    Args:
        policy: Any object implementing ``BasePolicy.infer()``.  If the
                policy also implements the ``TaskLifecycle`` protocol (has
                ``on_task_begin`` / ``on_task_end``), the server will call
                those methods at connection open / close.
        host: Bind address (default ``"0.0.0.0"`` — all interfaces).
        port: TCP port.  ``None`` lets the OS choose a free port.
        metadata: Arbitrary dict sent to the client immediately after the
                  WebSocket handshake.  Typically includes robot type, action
                  shape, etc.  Defaults to ``{}``.
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

        # Notify the policy that a new task (connection) is starting.
        # InferenceInterceptor implements on_task_begin(); plain Policy does not.
        # The hasattr check keeps this server decoupled from cache internals.
        if hasattr(self._policy, "on_task_begin"):
            self._policy.on_task_begin()

        await websocket.send(packer.pack(self._metadata))

        # prev_total_time tracks total round-trip time of the *previous*
        # request (infer + send) so it can be reported in the *next* response.
        # This gives the client visibility into end-to-end cycle time.
        prev_total_time = None

        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if "__ctrl__" in obs:
                    ctrl = obs["__ctrl__"]
                    if ctrl == "episode_start":
                        if hasattr(self._policy, "on_episode_start"):
                            self._policy.on_episode_start(
                                obs.get("__experiment__", "unknown"),
                                obs.get("__task__", ""),
                                obs.get("__episode_id__", -1),
                            )
                        await websocket.send(packer.pack({"__ack__": "episode_start"}))
                    elif ctrl == "episode_end":
                        if hasattr(self._policy, "on_episode_end"):
                            self._policy.on_episode_end(obs.get("__success__", False))
                        await websocket.send(packer.pack({"__ack__": "episode_end"}))
                    else:
                        await websocket.send(packer.pack({"__ack__": "ignored"}))
                    continue

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                # server_timing: wall-clock time spent in policy.infer(),
                # as measured by this server.  Does not include network IO.
                # For per-stage GPU timing, see SystemTimer output at task end.
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # Round-trip time of the previous cycle (infer + network send).
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                # Notify the policy that the task has ended.
                # SystemTimer.on_task_end() will print the per-probe timing
                # summary and (if configured) write a CSV file.
                if hasattr(self._policy, "on_task_end"):
                    self._policy.on_task_end()
                break

            except Exception:
                # Send the traceback to the client before closing so remote
                # debugging is possible without SSH access to the server.
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                # Ensure task-end hooks run even on error so timing records
                # are not silently lost (e.g. CSV is still flushed).
                if hasattr(self._policy, "on_task_end"):
                    self._policy.on_task_end()
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    """Respond to ``GET /healthz`` with 200 OK; pass other requests through."""
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal WebSocket handshake for all other paths.
    return None
