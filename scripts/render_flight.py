"""Flight GIF and trajectory plot of a trained policy.

    python scripts/render_flight.py --params runs/main/params_stage2.pkl --seed 3 --out assets/demo.gif
    python scripts/render_flight.py --params runs/main/params_stage2.pkl --topdown plots/trajectories.png --n 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hoopercargo.env import HooperCargoEnv  # noqa: E402
from hoopercargo.evaluate import load_policy  # noqa: E402
from make_plots import INK_2, SERIES  # noqa: E402

FRAME_EVERY = 3   # control steps per GIF frame (0.06 s)


def fly(params, seed: int, stage: int, scale: float, max_seconds: float | None = None):
    """One deterministic episode; returns the env and per-step pipeline states (host numpy)."""
    env = HooperCargoEnv(stage=stage, scale=scale)
    policy = jax.jit(load_policy(params))
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    steps = env.params.episode_steps if max_seconds is None else int(max_seconds / env.dt)
    trace = [state]
    for _ in range(steps):
        state = step(state, policy(state.obs))
        trace.append(state)
        if bool(state.done):
            break
    return env, trace


def render_flight(params, seed: int, out_gif, stage: int = 2, scale: float = 1.0,
                  max_seconds: float | None = None, width: int = 400, height: int = 300) -> Path:
    env, trace = fly(params, seed, stage, scale, max_seconds)
    m = env.host_model
    d = mujoco.MjData(m)
    payload_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    cam = mujoco.MjvCamera()
    cam.distance = 6.0 * scale
    cam.elevation = -15.0
    cam.azimuth = 135.0
    frames = []
    with mujoco.Renderer(m, height=height, width=width) as renderer:
        for state in trace[::FRAME_EVERY]:
            ps = state.pipeline_state
            d.qpos[:] = np.asarray(ps.data.qpos)
            d.mocap_pos[:] = np.asarray(ps.data.mocap_pos)
            m.geom_pos[payload_geom] = np.asarray(ps.cargo.offset)
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.subtree_com[env.ids.hopper]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_gif, np.stack(frames), duration=int(1000 * FRAME_EVERY * env.dt), loop=0)
    return out_gif


def plot_trajectories(params, n: int, seed: int, out_png, stage: int = 2, scale: float = 1.0,
                      max_seconds: float | None = None) -> Path:
    """Side and top views in a pad-aligned frame (pad on the +x axis)."""
    fig, (side, top) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    seen = set()
    for i in range(n):
        env, trace = fly(params, seed + i, stage, scale, max_seconds)
        hop = env.ids.hopper
        pad = np.asarray(trace[0].pipeline_state.pad_xy)
        ang = np.arctan2(pad[1], pad[0])
        rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
        com = np.array([np.asarray(s.pipeline_state.data.subtree_com[hop]) for s in trace])
        xy = com[:, :2] @ rot.T
        ok = float(trace[-1].metrics["success"]) > 0.5
        color, label = (SERIES[0], "safe landing") if ok else (SERIES[1], "failed")
        kw = dict(color=color, linewidth=1.2, alpha=0.85,
                  label=label if label not in seen else None)
        seen.add(label)
        side.plot(xy[:, 0], com[:, 2], **kw)
        top.plot(xy[:, 0], xy[:, 1], **kw)
        top.plot([np.linalg.norm(pad)], [0.0], marker="o", color=INK_2, markersize=4)
    side.set_ylabel("Altitude (m)")
    side.set_title(f"Flight paths, pad-aligned frame ({n} random flights)", loc="left")
    side.legend(frameon=False, loc="upper left")
    top.set_ylabel("Cross-track (m)")
    top.set_xlabel("Along-track distance toward the pad (m)")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, help="GIF path")
    ap.add_argument("--topdown", type=Path, help="trajectory plot path")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args(argv)
    if args.out:
        print(render_flight(args.params, args.seed, args.out, args.stage, args.scale))
    if args.topdown:
        print(plot_trajectories(args.params, args.n, args.seed, args.topdown, args.stage, args.scale))


if __name__ == "__main__":
    main()
