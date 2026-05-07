"""Request / response schema for the WebSocket policy protocol.

The protocol is binary msgpack with the numpy ndarray codec defined in
`server.msgpack_numpy`. Each connection follows this sequence:

    server  --[metadata: msgpack(dict)]-->                   client
    client  --[obs: msgpack(InferenceRequest)]-->            server
    server  --[result: msgpack(InferenceResponse)]-->        client
    ... (repeat infer cycle until client closes) ...

On error, the server sends a single TEXT frame with a traceback and
closes the connection with code 1011 (Internal Error). Clients should
treat any text frame as an error.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
from typing_extensions import NotRequired, TypedDict


class ServerMetadata(TypedDict, total=False):
    """First frame sent by the server immediately after a connection opens.

    All fields are optional — the server includes whatever the policy
    advertises. Clients should treat unknown keys as opaque.
    """

    protocol_version: str           # e.g. "1.0"
    policy_name: str                # human-readable name, e.g. "pi0-fast-base"
    control_mode: str               # e.g. "joints" / "end_pose" / "delta_eef"
    action_horizon: int             # number of action steps per chunk
    action_dim: int                 # length of each action vector
    state_dim: int                  # length of the state vector accepted in `state`
    image_keys: list                # list[str] of camera names the server expects
    image_shape: list               # [C, H, W] expected per image (CHW, uint8)
    expects_prompt: bool            # whether a language `prompt` is required
    extra: Dict[str, Any]           # free-form additional metadata


class InferenceRequest(TypedDict, total=False):
    """Single observation message from client → server.

        policy_input = {
            "images": {"cam_high": ndarray, ...},  # uint8 CHW arrays
            "state": ndarray,                       # 1-D float (state_dim,)
            "prompt": "pick up the cube",          # optional language goal
        }

    Cameras are keyed by name (the set is policy-specific; the server
    advertises the expected keys via metadata.image_keys).
    """

    images: Mapping[str, np.ndarray]                # camera_name -> uint8 (C,H,W)
    state: np.ndarray                               # float32/float64 (state_dim,)
    prompt: NotRequired[Optional[str]]              # natural-language instruction
    request_id: NotRequired[str]                    # echoed back for tracing


class ServerTiming(TypedDict, total=False):
    """Latency breakdown returned with every inference."""

    infer_ms: float                                 # policy.infer wallclock
    prev_total_ms: NotRequired[float]               # full RTT of the previous step


class InferenceResponse(TypedDict, total=False):
    """Server → client reply. `actions` is the only required key.

    Shape is (action_horizon, action_dim) — float32 ndarray. Clients are
    expected to consume the whole chunk or sub-sample as desired.
    """

    actions: np.ndarray                             # (action_horizon, action_dim)
    server_timing: ServerTiming
    request_id: NotRequired[str]                    # echoed from request, if provided
    extra: NotRequired[Dict[str, Any]]              # free-form, policy-specific
