# policy_deployment

A public-IP-facing **WebSocket server** for serving robot policies. Self-contained:
the only runtime dependencies are `websockets`, `msgpack`, and `numpy`.

The wire protocol is binary msgpack with a small numpy-ndarray extension
(`server/msgpack_numpy.py`). The shape of the `metadata` / observation /
response messages is defined explicitly in `server/schema.py`, so any
client that speaks msgpack over a WebSocket can talk to this server.

## Repository layout

```
server/
  __init__.py
  schema.py              # InferenceRequest / InferenceResponse / ServerMetadata (typed)
  policy.py              # BasePolicy: implement metadata + infer()
  websocket_server.py    # WebsocketPolicyServer (asyncio)
  auth.py                # API key + IP allowlist for public IP
  msgpack_numpy.py       # msgpack codec with ndarray support
examples/
  echo_policy.py         # smoke-test policy (no model required)
  my_policy.py           # template for plugging in your own model
scripts/
  launch.py              # CLI entrypoint
  ping.py                # handshake / metadata check
  smoke_test.py          # end-to-end inference check
  _client.py             # tiny in-repo client used by the test scripts
deploy/
  nginx.conf.example     # TLS termination + reverse proxy
  policy-server.service  # systemd unit
```

## Protocol

Binary WebSocket frames, msgpack with the numpy extension defined in
`server/msgpack_numpy.py`. Each connection runs:

```
server  ── metadata (msgpack dict) ──▶ client          (immediately on connect)
client  ── observation (msgpack)  ──▶ server
server  ── result (msgpack)       ──▶ client           (one per observation)
... infer loop continues until the client closes ...
```

### Server → client metadata (first frame)

Defined by `server.schema.ServerMetadata`:

| key                | type     | meaning                                          |
|--------------------|----------|--------------------------------------------------|
| `protocol_version` | str      | e.g. `"1.0"`                                     |
| `policy_name`      | str      | human-readable name                              |
| `control_mode`     | str      | `"joints"`, `"end_pose"`, `"delta_eef"`, ...     |
| `action_horizon`   | int      | timesteps in each returned action chunk          |
| `action_dim`       | int      | length of each action vector                     |
| `state_dim`        | int      | length of `state` accepted in observations       |
| `image_keys`       | list[str]| camera names the server expects                  |
| `image_shape`      | list[int]| `[C, H, W]` per image (CHW, uint8)               |
| `expects_prompt`   | bool     | whether a language `prompt` is required          |
| `extra`            | dict     | free-form, policy-specific                       |

### Client → server observation

Defined by `server.schema.InferenceRequest`:

```python
{
    "images": {                                  # dict[camera_name -> ndarray]
        "cam_high":        np.ndarray (C,H,W) uint8,
        "cam_left_wrist":  np.ndarray (C,H,W) uint8,
        "cam_right_wrist": np.ndarray (C,H,W) uint8,
    },
    "state": np.ndarray (state_dim,) float32/64,
    "prompt": "pick up the cube",                # optional
    "request_id": "abc-123",                     # optional, echoed in response
}
```

Camera images are CHW uint8 arrays. The exact set of expected camera
keys is policy-specific and is advertised via `metadata.image_keys`.

### Server → client response

Defined by `server.schema.InferenceResponse`:

```python
{
    "actions":       np.ndarray (action_horizon, action_dim) float32,
    "server_timing": {
        "infer_ms":      <float>,
        "prev_total_ms": <float>,                # full RTT of the previous step
    },
    "request_id": "abc-123",                     # echoed if provided
}
```

The shape is **always** `(action_horizon, action_dim)`. Clients consume
the chunk by indexing rows of `result["actions"]`.

### Errors

If `policy.infer` raises, the server sends a single **text** frame with
the traceback, then closes the connection with code 1011 (Internal Error).
Clients should treat any text frame as an error.

## Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Local smoke test with the echo policy
PYTHONPATH=. python scripts/launch.py --policy examples.echo_policy:EchoPolicy

# in another shell:
PYTHONPATH=. python scripts/ping.py       --host 127.0.0.1 --port 8000
PYTHONPATH=. python scripts/smoke_test.py --host 127.0.0.1 --port 8000
```

## Plugging in your model

Edit `examples/my_policy.py` (or copy it):

```python
from server.policy import BasePolicy
from server.schema import InferenceResponse, ServerMetadata

class MyPolicy(BasePolicy):
    @property
    def metadata(self) -> ServerMetadata:
        return {"protocol_version": "1.0", "action_horizon": 10,
                "action_dim": 14, "state_dim": 14, ...}

    def infer(self, obs) -> InferenceResponse:
        actions = self.model(obs["images"], obs["state"], obs.get("prompt"))
        return {"actions": actions}                  # shape (T, action_dim)
```

Launch it:

```bash
PYTHONPATH=. python scripts/launch.py \
    --policy examples.my_policy:MyPolicy \
    --policy-kwargs checkpoint_path=/data/ckpt.pt \
    --policy-kwargs device=cuda:0
```

## Public-IP deployment

The server is intended to listen on a public address. Defaults are tuned
for that case, but **always** add at least these protections:

1. **API key.** Set `POLICY_SERVER_API_KEYS=key1,key2` (or pass `--api-key`
   one or more times). Clients authenticate with
   `Authorization: Api-Key <KEY>`.

2. **IP allowlist (optional but recommended).** Pass `--allow-cidr` one or
   more times. The default empty list allows all source IPs.

3. **Connection cap.** `--max-connections` (default 16) bounds concurrency
   per process; extra clients get HTTP 503.

4. **Per-frame size limit.** `--max-message-size` (default 16 MiB) caps the
   biggest observation. Set to your largest realistic image payload.

5. **TLS.** The server speaks plain `ws://`. For `wss://` put it behind
   nginx / Caddy / a cloud LB (`deploy/nginx.conf.example` shows the
   minimum nginx config).

6. **Process supervision.** Use `deploy/policy-server.service` (systemd) or
   any equivalent supervisor.

Health check: `GET /healthz` returns `200 OK` (used by load balancers /
k8s probes). It does **not** require an API key.

## Notes

- The server forwards `request_id` from observation to response if the
  client sets one — convenient for tracing across a load balancer.
- `scripts/_client.py` is a minimal client kept for the test scripts;
  the protocol is documented above so production callers can implement
  their own in any language.
