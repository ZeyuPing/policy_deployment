#!/usr/bin/env python3
"""Launch the WebSocket policy server.

Examples:

    # Local dev with the echo policy:
    PYTHONPATH=. python scripts/launch.py \\
        --policy examples.echo_policy:EchoPolicy

    # Production: bind public IP, require API key, restrict source IPs:
    POLICY_SERVER_API_KEYS=$(openssl rand -hex 32) \\
    PYTHONPATH=. python scripts/launch.py \\
        --policy examples.my_policy:MyPolicy \\
        --policy-kwargs checkpoint_path=/data/ckpt.pt \\
        --host 0.0.0.0 --port 8000 \\
        --allow-cidr 10.0.0.0/8 --allow-cidr 192.168.1.0/24 \\
        --max-connections 8
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from typing import Any, Dict, List

from server.auth import AuthConfig
from server.websocket_server import WebsocketPolicyServer


def _import_policy(spec: str):
    """Resolve a 'module.path:ClassName' string to a class object."""
    if ":" not in spec:
        raise ValueError(
            f"--policy must be 'module.path:ClassName' (got {spec!r})"
        )
    mod_name, cls_name = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, cls_name)


def _parse_kv_list(items: List[str]) -> Dict[str, Any]:
    """Parse ['k=v', 'k2=v2'] into a dict; coerces ints/floats/bools."""
    out: Dict[str, Any] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"--policy-kwargs entry must be k=v, got {raw!r}")
        k, v = raw.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Public-IP-facing WebSocket policy server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--policy", required=True,
        help="Import path of the policy class, e.g. examples.echo_policy:EchoPolicy",
    )
    p.add_argument(
        "--policy-kwargs", action="append", default=[],
        metavar="K=V", help="Constructor kwargs for the policy (repeatable).",
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="Bind address. Use 0.0.0.0 to accept public IPs.")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", action="append", default=[],
                   help="Accept this API key (repeatable). "
                        "Also reads POLICY_SERVER_API_KEYS env var (comma list).")
    p.add_argument("--allow-cidr", action="append", default=[],
                   help="Restrict source IPs to these CIDR blocks (repeatable).")
    p.add_argument("--max-connections", type=int, default=16)
    p.add_argument("--max-message-size", type=int, default=16 * 1024 * 1024,
                   help="Per-frame size cap in bytes (0 = unlimited).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("launch")

    cls = _import_policy(args.policy)
    kwargs = _parse_kv_list(args.policy_kwargs)
    logger.info("Instantiating policy %s with %s", args.policy, kwargs)
    policy = cls(**kwargs)

    auth = AuthConfig.from_env_and_args(
        api_keys=args.api_key,
        ip_allowlist=args.allow_cidr,
    )
    if not auth.api_keys and args.host == "0.0.0.0":
        logger.warning(
            "Server is bound to 0.0.0.0 but NO API key is configured. "
            "Public deployments should set --api-key or POLICY_SERVER_API_KEYS."
        )

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        auth=auth,
        max_connections=args.max_connections,
        max_message_size=args.max_message_size or None,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopped (Ctrl+C)")
        sys.exit(0)


if __name__ == "__main__":
    main()
