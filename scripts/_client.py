"""Minimal synchronous WebSocket client used by the test scripts.

This is intentionally tiny — production clients should bring their own
implementation. It speaks the same protocol as `server.websocket_server`:
msgpack-numpy frames, metadata first, then obs/result round-trips.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from server import msgpack_numpy


class WebsocketPolicyClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        connect_retry_sec: float = 5.0,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._connect(connect_retry_sec)

    def _connect(self, retry_sec: float) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        logging.info("Connecting to %s ...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("server not ready, retrying in %.1fs", retry_sec)
                time.sleep(retry_sec)

    @property
    def metadata(self) -> Dict:
        return self._metadata

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass
