"""Template — copy and edit to plug in your own model.

Run with::

    PYTHONPATH=. python scripts/launch.py \\
        --policy examples.my_policy:MyPolicy \\
        --checkpoint /path/to/ckpt --port 8000
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from server.policy import BasePolicy
from server.schema import InferenceResponse, ServerMetadata


class MyPolicy(BasePolicy):
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        action_horizon: int = 10,
        action_dim: int = 14,
        state_dim: int = 14,
        control_mode: str = "joints",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.control_mode = control_mode
        self.model = self._load_model(checkpoint_path, device)

    def _load_model(self, checkpoint_path: str, device: str):
        # TODO: load your weights here
        raise NotImplementedError("Implement _load_model() before serving")

    @property
    def metadata(self) -> ServerMetadata:
        return {
            "protocol_version": "1.0",
            "policy_name": "my_policy",
            "control_mode": self.control_mode,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "image_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            "image_shape": [3, 224, 224],
            "expects_prompt": True,
        }

    def infer(self, obs: Dict[str, Any]) -> InferenceResponse:
        # obs["images"] is dict[str, np.ndarray (C,H,W) uint8]
        # obs["state"]  is np.ndarray (state_dim,)
        # obs.get("prompt") is Optional[str]
        # TODO: build your model input, run forward, return chunk shape (T, action_dim)
        actions = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        return {"actions": actions}
