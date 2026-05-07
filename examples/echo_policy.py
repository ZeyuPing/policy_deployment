"""Reference / smoke-test policy.

Returns a flat action chunk that simply tiles the incoming state. Useful
to verify the wire protocol end-to-end without loading any real model.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from server.policy import BasePolicy
from server.schema import InferenceResponse, ServerMetadata


class EchoPolicy(BasePolicy):
    def __init__(
        self,
        action_horizon: int = 10,
        action_dim: int = 14,
        state_dim: int = 14,
        image_keys: tuple = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        image_shape: tuple = (3, 224, 224),
        control_mode: str = "joints",
    ) -> None:
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_keys = list(image_keys)
        self.image_shape = list(image_shape)
        self.control_mode = control_mode

    @property
    def metadata(self) -> ServerMetadata:
        return {
            "protocol_version": "1.0",
            "policy_name": "echo",
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "image_keys": self.image_keys,
            "image_shape": self.image_shape,
            "expects_prompt": False,
        }

    def infer(self, obs: Dict[str, Any]) -> InferenceResponse:
        state = np.asarray(obs.get("state", np.zeros(self.action_dim)), dtype=np.float32)
        if state.size != self.action_dim:
            # Pad / truncate so we always return a (T, action_dim) chunk.
            buf = np.zeros(self.action_dim, dtype=np.float32)
            n = min(state.size, self.action_dim)
            buf[:n] = state.flatten()[:n]
            state = buf
        actions = np.tile(state[None, :], (self.action_horizon, 1)).astype(np.float32)
        return {"actions": actions}
