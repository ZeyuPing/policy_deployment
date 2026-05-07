"""BasePolicy interface — subclass this to plug in your model."""

from __future__ import annotations

import abc
from typing import Any, Dict

from server.schema import InferenceRequest, InferenceResponse, ServerMetadata


class BasePolicy(abc.ABC):
    """Server-side policy contract.

    Implementations must populate `metadata` (used for the handshake) and
    implement `infer(obs)`. The server enforces wire format; this class
    works in pure Python / numpy types.
    """

    @property
    @abc.abstractmethod
    def metadata(self) -> ServerMetadata:
        """Returned to the client as the first frame after connecting.

        Must be a plain dict that msgpack can encode (no numpy arrays).
        """

    @abc.abstractmethod
    def infer(self, obs: InferenceRequest) -> InferenceResponse:
        """Run a single inference step.

        Args:
            obs: A dict with keys defined by `InferenceRequest`. Numpy
                 arrays in `images` and `state` are already deserialised.

        Returns:
            A dict with at least `{"actions": np.ndarray}` of shape
            (action_horizon, action_dim). The server adds `server_timing`
            before sending.
        """

    def reset(self) -> None:
        """Optional hook called when a new client connects."""
        return None

    def close(self) -> None:
        """Optional hook called on server shutdown."""
        return None
