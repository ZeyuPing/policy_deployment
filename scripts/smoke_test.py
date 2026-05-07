#!/usr/bin/env python3
"""End-to-end smoke test: connect, send one observation, validate the action chunk."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scripts._client import WebsocketPolicyClient


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=None)
    args = p.parse_args()

    client = WebsocketPolicyClient(host=args.host, port=args.port, api_key=args.api_key)
    meta = client.metadata
    print(f"metadata: {meta}")

    image_keys = meta.get("image_keys") or ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    image_shape = tuple(meta.get("image_shape") or (3, 224, 224))
    state_dim = int(meta.get("state_dim", 14))

    obs = {
        "images": {k: np.zeros(image_shape, dtype=np.uint8) for k in image_keys},
        "state": np.zeros(state_dim, dtype=np.float32),
        "prompt": "smoke test",
    }
    result = client.infer(obs)

    if "actions" not in result:
        print(f"[FAIL] missing 'actions' in response (keys={list(result.keys())})")
        return 1
    actions = np.asarray(result["actions"])
    if actions.ndim != 2:
        print(f"[FAIL] actions must be 2-D, got shape {actions.shape}")
        return 1
    print(f"[PASS] actions shape={actions.shape} dtype={actions.dtype}")
    print(f"server_timing: {result.get('server_timing')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
