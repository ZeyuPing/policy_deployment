#!/usr/bin/env python3
"""Ping / handshake checker for the policy server.

Verifies:
  1. WebSocket connection succeeds (auth, allowlist, etc.).
  2. First server frame is msgpack-encoded metadata (not a text error).
  3. Required metadata fields are present and have sane types.
"""

from __future__ import annotations

import argparse
import json
import sys

from scripts._client import WebsocketPolicyClient


REQUIRED = {
    "protocol_version": str,
    "action_horizon": int,
    "action_dim": int,
    "state_dim": int,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    try:
        client = WebsocketPolicyClient(
            host=args.host, port=args.port, api_key=args.api_key
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] connect: {exc}")
        return 1

    meta = client.metadata
    if not isinstance(meta, dict):
        print(f"[FAIL] metadata is not a dict: {type(meta).__name__}")
        return 1

    errors = []
    for key, typ in REQUIRED.items():
        if key not in meta:
            errors.append(f"missing key: {key}")
        elif not isinstance(meta[key], typ):
            errors.append(
                f"bad type for {key}: expected {typ.__name__}, "
                f"got {type(meta[key]).__name__}"
            )

    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    if errors:
        print("[FAIL]")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[PASS] handshake OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
