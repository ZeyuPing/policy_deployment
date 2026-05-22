# Sim check (`sim/`)

A small MuJoCo dual-arm scene + a handful of recorded demos. Use it to
sanity-check that your policy server produces reasonable bimanual
behavior **before** running a full eval.

The recommended workflow is:

1. `--mode replay` — confirm the recorded demo looks right (scene loads,
   trajectory is sensible). No policy needed.
2. `--mode policy` — let your policy drive the sim from scratch.
3. `--mode compare` — for each sample, run the recorded chunk and the
   policy's predicted chunk back-to-back from the same starting state.
   This is the **main verification mode**.

## Contents

```
sim/
  check_in_sim.py                 # entry point (replay / policy / compare)
  assets/
    example_slim.pkl              # 1-shot example bundle (small)
    robot_models/                 # MuJoCo XML + meshes for dual-YAM
```

The default scene is
`assets/robot_models/arm/dual_yam/dual_yam_bimanual.xml`. Actuator order
(left arm 1-6, left gripper, right arm 1-6, right gripper) matches the
14-dim action layout below.

## Requirements

```
pip install mujoco numpy pillow
pip install imageio                 # only for --output FILE.mp4 (offscreen video)
```

macOS only: the live MuJoCo viewer needs the main thread, so launch with
`mjpython` (ships with the `mujoco` wheel) instead of `python`. Plain
`python` works for `--output FILE.mp4` on every platform.

## Action / state convention (14-dim)

Both the recorded actions and what your policy must return follow this
layout — **same units everywhere**:

```
[0:6]   left  arm joints 1-6   (radians)
[6]     left  gripper          (normalized: 0.0 = closed, 1.0 = open)
[7:13]  right arm joints 1-6   (radians)
[13]    right gripper          (normalized)
```

Internally the sim maps the normalized gripper value onto MuJoCo
ctrlrange `[0.0, 0.0475]` (≈ 47.5 mm jaw opening). Your server should
output gripper values in `[0, 1]`, not raw mm.

## Quick start

### 1. Replay the recorded demo (no policy)

```bash
mjpython sim/check_in_sim.py --mode replay \
    --bundle sim/assets/example_slim.pkl
```

If the arms move toward the object and the grippers open/close at the
right moments, the scene and bundle are healthy.

Offscreen mp4 (no window, any platform):

```bash
python sim/check_in_sim.py --mode replay \
    --bundle sim/assets/example_slim.pkl \
    --output out/replay.mp4 --render-camera front
```

### 2. Drive the sim from your policy server

Start your server (see top-level `README.md` → `scripts/launch.py`),
then:

```bash
mjpython sim/check_in_sim.py --mode policy \
    --bundle sim/assets/example_slim.pkl \
    --host 127.0.0.1 --port 8000 \
    --prompt "Insert the battery to the mouse." \
    --action-horizon 50
```

At each step the script builds an obs as documented in the top-level
README (`images` dict of CHW uint8 arrays, `state` 14-dim float, optional
`prompt`), pushes it to the server, and executes the returned chunk.

### 3. Compare recorded vs predicted (**main check**)

```bash
mjpython sim/check_in_sim.py --mode compare \
    --bundle sim/assets/example_slim.pkl \
    --host 127.0.0.1 --port 8000 \
    --prompt "Insert the battery to the mouse." \
    --action-horizon 50
```

For each sample in the slim bundle the script will:

1. Snap qpos + ctrl to the recorded starting state, then execute the
   **recorded** action chunk. You see what the demonstration looks like.
2. Snap back to the same starting state, query the server, then execute
   the **predicted** chunk. You see what your policy would do from the
   same state.
3. Print per-sample diff metrics:

   ```
   sample[ 0]  diff over 30 steps:  L2=0.1234  L2/sqrt(N*14)=0.0067  max|.|=0.0420
               max|.| per dim: 0.005 0.011 ... 0.034
   ```

   `L2/sqrt(N*14)` is the average per-element error; `max|.| per dim`
   tells you which joint diverges most.

## How to read the compare output

| Symptom                                            | Likely cause                                                                 |
|----------------------------------------------------|------------------------------------------------------------------------------|
| Big diff only on dim `6` and/or `13`               | Gripper convention wrong — must be normalized `[0, 1]`, not mm or `[-2.7, 0]`.|
| Big diff on arm joints, similar magnitude on each  | Joints in wrong order, or degrees instead of radians.                        |
| Left and right swapped visually                    | Action layout is `[L_arm, L_grip, R_arm, R_grip]`, not interleaved.          |
| Predicted rollout drifts off-screen                | State scale off (proprio probably in a different frame than what was trained on). |
| Recorded looks fine, predicted doesn't move at all | Server is returning a constant action chunk — check that your policy actually uses the obs. |

A reasonable policy on the example bundle should produce visually
similar end poses and `L2/sqrt(N*14)` under ~0.05 on the arm dims. Some
divergence is fine — these are stochastic demos, not a single ground
truth.

## Slim bundle format

`assets/example_slim.pkl` is a pickled dict:

```python
{
  "meta":   {"n_frames": int, "duration_s": float, ...},

  # --- used by --mode replay ---
  "actions":       ndarray (T, 14) float64,    # 14-dim action stream
  "joint_positions": ndarray (T, 14) float64,  # optional, --source override

  # --- used by --mode policy ---
  "timestamps_ns": ndarray (T,) int64,
  "images": {
      "top_camera":   list[bytes],             # JPEG bytes per frame
      "left_camera":  list[bytes],
      "right_camera": list[bytes],
  },

  # --- used by --mode compare ---
  "samples": [
      {
          "frame_idx":    int,
          "state":        ndarray (14,) float64,
          "action_chunk": ndarray (H, 14) float64,
          "images": {
              "top_camera":   bytes,           # one JPEG per sample
              "left_camera":  bytes,
              "right_camera": bytes,
          },
      },
      ...
  ],
}
```

The example bundle has all fields populated, so all three modes work
out of the box.

## All command-line options

| Flag                 | Default                                           | Meaning                                                  |
|----------------------|---------------------------------------------------|----------------------------------------------------------|
| `--mode`             | (required) `replay` / `policy` / `compare`        | Run mode                                                 |
| `--bundle`           | `sim/assets/example_slim.pkl`                     | Input bundle                                             |
| `--scene`            | `…/dual_yam/dual_yam_bimanual.xml`                | MuJoCo XML                                               |
| `--ctrl-hz`          | `60.0`                                            | Control loop frequency                                   |
| `--speed`            | `1.0`                                             | Wallclock playback multiplier (viewer mode only)         |
| `--source`           | `actions`                                         | `replay` only: which 14-dim stream to play               |
| `--host` / `--port`  | `127.0.0.1` / `8000`                              | `policy` / `compare`: policy server endpoint             |
| `--api-key`          | unset                                             | `policy` / `compare`: API key if the server requires one |
| `--prompt`           | `""`                                              | Language goal forwarded to the server                    |
| `--action-horizon`   | `10`                                              | Actions per policy chunk to execute before re-querying   |
| `--output FILE.mp4`  | off                                               | Render offscreen to mp4 (no viewer; deterministic)       |
| `--render-camera`    | `front`                                           | Named scene camera (`front`, `top`, `side_left`)         |
| `--render-width/-height` | `640` / `480`                                 | Output frame size                                        |
| `--fps`              | `30`                                              | Output video framerate                                   |
| `--pause-s`          | `0.5`                                             | `compare` viewer only: pause between recorded / policy   |

## Troubleshooting

- **macOS: viewer window won't open / GLFW error** → use `mjpython` for
  viewer mode, or render to mp4 with `--output`.
- **`RuntimeError: Joint left_joint1 not in scene`** → wrong `--scene`;
  the bundled XML is the only one that matches the 14-dim layout.
- **Policy mode hangs at connect** → server isn't up, or you're hitting
  the wrong host/port. If the server requires an API key, pass it with
  `--api-key`.
- **`compare` mode reports `requires a slim bundle`** → your pkl has
  no `"samples"` key; regenerate the bundle or use the example.
