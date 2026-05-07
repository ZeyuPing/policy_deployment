"""WebSocket policy server.

The on-the-wire protocol is binary msgpack with the numpy ndarray codec
defined in `server/msgpack_numpy.py`. See `server/schema.py` for the
message shapes.
"""

from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback
from typing import Optional

import websockets.asyncio.server as _ws_server
import websockets.frames

from server import msgpack_numpy
from server.auth import AuthConfig, check_request
from server.policy import BasePolicy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Public-IP-facing policy server over WebSocket.

    Args:
        policy:           A BasePolicy implementation.
        host:             Bind address. Use "0.0.0.0" to accept public IPs.
        port:             TCP port to listen on.
        auth:             AuthConfig with API keys / IP allowlist.
        max_connections:  Cap on concurrent active clients (extra clients are
                          immediately rejected with 503 to limit DoS surface).
        max_message_size: Max bytes per inbound frame. None = unlimited.
                          Public-facing servers should set this (e.g. 16 MiB).
        ping_interval:    Seconds between server-initiated pings (keepalive).
        ping_timeout:     Seconds to wait for pong before dropping connection.
    """

    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        auth: Optional[AuthConfig] = None,
        max_connections: int = 16,
        max_message_size: Optional[int] = 16 * 1024 * 1024,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._auth = auth or AuthConfig()
        self._max_connections = max_connections
        self._max_message_size = max_message_size
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._active = 0
        self._active_lock = asyncio.Lock()

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    # ── Public API ────────────────────────────────────────────────

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        finally:
            try:
                self._policy.close()
            except Exception:  # noqa: BLE001
                logger.exception("policy.close() raised")

    async def run(self) -> None:
        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=self._max_message_size,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            process_request=self._process_request,
        ) as server:
            logger.info(
                "WebSocketPolicyServer listening on ws://%s:%d "
                "(auth=%s, ip_allowlist=%s, max_conn=%d)",
                self._host,
                self._port,
                "on" if self._auth.api_keys else "off",
                len(self._auth.ip_allowlist) or "all",
                self._max_connections,
            )
            await server.serve_forever()

    # ── Request validation (runs before WS upgrade) ───────────────

    async def _process_request(
        self,
        connection: _ws_server.ServerConnection,
        request: _ws_server.Request,
    ):
        # Health-check endpoint for load balancers / k8s probes.
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")

        # Auth & IP allowlist.
        remote = connection.remote_address
        rejection = check_request(self._auth, remote, request.headers)
        if rejection is not None:
            status, body = rejection
            return connection.respond(status, body)

        # Connection cap (protects against accidental fan-out / DoS).
        if self._active >= self._max_connections:
            logger.warning(
                "Rejecting %s: connection limit reached (%d)",
                remote,
                self._max_connections,
            )
            return connection.respond(
                http.HTTPStatus.SERVICE_UNAVAILABLE,
                "Server at capacity\n",
            )
        return None

    # ── Per-connection handler ────────────────────────────────────

    async def _handler(self, websocket: _ws_server.ServerConnection) -> None:
        async with self._active_lock:
            self._active += 1
        remote = websocket.remote_address
        logger.info("Connection opened from %s (active=%d)", remote, self._active)

        try:
            try:
                self._policy.reset()
            except Exception:  # noqa: BLE001
                logger.exception("policy.reset() raised")

            packer = msgpack_numpy.Packer()
            await websocket.send(packer.pack(self._policy.metadata))

            prev_total_time: Optional[float] = None
            while True:
                try:
                    start = time.monotonic()
                    raw = await websocket.recv()
                    if isinstance(raw, str):
                        # Clients send only binary frames; ignore stray text.
                        logger.warning("Ignoring text frame from %s", remote)
                        continue
                    obs = msgpack_numpy.unpackb(raw)

                    infer_start = time.monotonic()
                    action = self._policy.infer(obs)
                    infer_ms = (time.monotonic() - infer_start) * 1000

                    if not isinstance(action, dict) or "actions" not in action:
                        raise RuntimeError(
                            "policy.infer must return dict with 'actions' key"
                        )

                    timing = action.setdefault("server_timing", {})
                    timing["infer_ms"] = infer_ms
                    if prev_total_time is not None:
                        timing["prev_total_ms"] = prev_total_time * 1000

                    if "request_id" in obs and "request_id" not in action:
                        action["request_id"] = obs["request_id"]

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start

                except websockets.ConnectionClosed:
                    logger.info("Connection from %s closed", remote)
                    break
                except Exception:  # noqa: BLE001
                    tb = traceback.format_exc()
                    logger.exception("Inference error for %s", remote)
                    try:
                        await websocket.send(tb)
                        await websocket.close(
                            code=websockets.frames.CloseCode.INTERNAL_ERROR,
                            reason="Internal server error.",
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    break
        finally:
            async with self._active_lock:
                self._active -= 1
            logger.info("Connection from %s ended (active=%d)", remote, self._active)
