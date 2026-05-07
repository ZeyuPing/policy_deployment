"""Authentication and access control for the public-facing server.

Two layers:

1. Static API key (header: ``Authorization: Api-Key <KEY>``). Multiple
   keys may be configured by setting POLICY_SERVER_API_KEYS to a
   comma-separated list, or by passing --api-key one or more times.

2. Optional CIDR-based IP allowlist. If provided, only matching client
   addresses are accepted. Empty allowlist = allow all.
"""

from __future__ import annotations

import http
import ipaddress
import logging
import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthConfig:
    """Server authentication settings.

    Attributes:
        api_keys: Set of accepted API keys. Empty = no auth required
                  (NOT recommended for public deployments).
        ip_allowlist: List of CIDR networks. Empty = allow all source IPs.
    """

    api_keys: frozenset = field(default_factory=frozenset)
    ip_allowlist: tuple = ()  # tuple[ipaddress._BaseNetwork, ...]

    @classmethod
    def from_env_and_args(
        cls,
        api_keys: Optional[Iterable[str]] = None,
        ip_allowlist: Optional[Iterable[str]] = None,
    ) -> "AuthConfig":
        keys: List[str] = []
        if api_keys:
            keys.extend(k.strip() for k in api_keys if k and k.strip())
        env_keys = os.environ.get("POLICY_SERVER_API_KEYS", "")
        if env_keys:
            keys.extend(k.strip() for k in env_keys.split(",") if k.strip())

        nets: List[ipaddress._BaseNetwork] = []
        for raw in ip_allowlist or ():
            raw = raw.strip()
            if not raw:
                continue
            nets.append(ipaddress.ip_network(raw, strict=False))

        return cls(api_keys=frozenset(keys), ip_allowlist=tuple(nets))


def _extract_api_key(headers) -> Optional[str]:
    """Pull the API key out of an Authorization header.

    Accepts both `Api-Key <key>` and `Bearer <key>` for convenience.
    Header lookup is case-insensitive (websockets uses CaseInsensitiveDict).
    """
    auth = headers.get("Authorization") or headers.get("authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2:
        return None
    scheme, value = parts[0].strip().lower(), parts[1].strip()
    if scheme not in ("api-key", "bearer"):
        return None
    return value or None


def check_request(
    cfg: AuthConfig,
    remote_addr: Optional[Tuple[str, int]],
    headers,
) -> Optional[Tuple[http.HTTPStatus, str]]:
    """Validate an incoming WebSocket upgrade request.

    Returns None on success, or (status, body) describing the rejection.
    """
    # 1. IP allowlist
    if cfg.ip_allowlist and remote_addr is not None:
        try:
            ip = ipaddress.ip_address(remote_addr[0])
        except ValueError:
            return http.HTTPStatus.FORBIDDEN, "Invalid client address\n"
        if not any(ip in net for net in cfg.ip_allowlist):
            logger.warning("Rejected connection from %s (not in allowlist)", ip)
            return http.HTTPStatus.FORBIDDEN, "IP not allowed\n"

    # 2. API key
    if cfg.api_keys:
        key = _extract_api_key(headers)
        if key is None:
            return http.HTTPStatus.UNAUTHORIZED, "Missing API key\n"
        if key not in cfg.api_keys:
            logger.warning("Rejected connection from %s (bad API key)", remote_addr)
            return http.HTTPStatus.UNAUTHORIZED, "Invalid API key\n"

    return None
