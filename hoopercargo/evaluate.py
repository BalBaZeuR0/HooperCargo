"""Deterministic evaluation of a trained policy over N random cargo flights -> CSV.

    python -m hoopercargo.evaluate --params runs/main/params_stage2.pkl --episodes 1000 --out results/eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
import jax
import jax.numpy as jp

from .env import OBS_SIZE, HooperCargoEnv
from .payload import Cargo, loaded_com

COLUMNS = ("episode", "payload_mass", "payload_z", "payload_lateral", "com_shift", "wind_mean",
           "pad_distance", "success", "fail_reason", "landing_error", "fuel_used", "flight_time",
           "touch_vz", "touch_vxy")
FAIL_KEYS = ("fail_tilt", "fail_offpad", "fail_hard", "fail_timeout")


def run_config(params_path: Path) -> dict:
    return json.loads((Path(params_path).parent / "config.json").read_text())


def make_eval_env(params_path: Path, stage: int, scale: float | None = None) -> HooperCargoEnv:
    """Evaluation env matching the run: its scale, and payload observed only if it was in training.

    Payloads are always random at evaluation, also for a fixed-payload baseline.
    """
    cfg = run_config(params_path)
    if scale is not None and float(scale) != float(cfg["scale"]):
        raise ValueError(f"scale {scale} does not match the run's scale {cfg['scale']}")
    return HooperCargoEnv(stage=stage, scale=cfg["scale"], randomize_payload=True,
                          observe_payload=cfg["randomize_payload"])


def load_policy(params_path: Path):
    """Deterministic policy `obs -> action` from a params file and its run's config.json."""
    params_path = Path(params_path)
    cfg = run_config(params_path)["ppo"]
    nets = ppo_networks.make_ppo_networks(
        OBS_SIZE, 3, preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=tuple(cfg["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(cfg["value_hidden_layer_sizes"]))
    params = brax_model.load_params(str(params_path))
    policy = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    return lambda obs: policy(obs, jax.random.PRNGKey(0))[0]


def _rollout_fn(env: HooperCargoEnv, policy):
    """Runs one batch of episodes to the end; returns reset states and final metrics."""
    def run(keys):
        start = jax.vmap(env.reset)(keys)

        def body(carry, _):
            state, final, finished = carry
            state = jax.vmap(env.step)(state, jax.vmap(policy)(state.obs))
            newly = (state.done > 0) & ~finished
            final = jax.tree.map(lambda f, m: jp.where(newly, m, f), final, state.metrics)
            return (state, final, finished | newly), None

        final0 = jax.tree.map(jp.zeros_like, start.metrics)
        (_, final, _), _ = jax.lax.scan(body, (start, final0, jp.zeros(keys.shape[0], bool)),
                                        None, length=env.params.episode_steps)
        return start, final
    return jax.jit(run)


def evaluate(params_path, episodes: int, seed: int, stage: int, scale: float | None, out_csv,
             batch: int = 250) -> list[dict]:
    env = make_eval_env(params_path, stage, scale)
    p = env.params
    run = _rollout_fn(env, load_policy(params_path))
    keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
    empty = Cargo(mass=jp.array(0.0), offset=jp.zeros(3))
    rows = []
    for lo in range(0, episodes, batch):
        start, final = run(keys[lo:lo + batch])
        ps = start.pipeline_state
        for i in range(keys[lo:lo + batch].shape[0]):
            cargo = jax.tree.map(lambda x: x[i], ps.cargo)
            shift = jp.linalg.norm(loaded_com(p, cargo, p.fuel_mass0) - loaded_com(p, empty, p.fuel_mass0))
            m = {k: float(v[i]) for k, v in final.items()}
            reason = next((k.removeprefix("fail_") for k in FAIL_KEYS if m[k] > 0.5), "none")
            rows.append(dict(
                episode=lo + i,
                payload_mass=float(cargo.mass),
                payload_z=float(cargo.offset[2]),
                payload_lateral=float(jp.linalg.norm(cargo.offset[:2])),
                com_shift=float(shift),
                wind_mean=float(jp.linalg.norm(ps.wind.mean[i])),
                pad_distance=float(jp.linalg.norm(ps.pad_xy[i])),
                success=int(m["success"] > 0.5),
                fail_reason=reason,
                landing_error=m["landing_error"],
                fuel_used=m["fuel_used"],
                flight_time=m["flight_time"],
                touch_vz=m["touch_vz"],
                touch_vxy=m["touch_vxy"],
            ))
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def summarize(rows: list[dict]) -> str:
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    reasons = {k: sum(r["fail_reason"] == k for r in rows) for k in ("tilt", "offpad", "hard", "timeout")}
    err = sum(r["landing_error"] for r in ok) / max(len(ok), 1)
    fuel = sum(r["fuel_used"] for r in ok) / max(len(ok), 1)
    return (f"safe landings: {len(ok)}/{n} ({100 * len(ok) / n:.1f}%) | "
            f"landing error {err:.2f} m, fuel used {100 * fuel:.0f}% (successful flights) | "
            f"failures {reasons}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--scale", type=float, default=None, help="optional check against the run's scale")
    ap.add_argument("--out", type=Path, default=Path("results/eval.csv"))
    args = ap.parse_args(argv)
    rows = evaluate(args.params, args.episodes, args.seed, args.stage, args.scale, args.out)
    print(summarize(rows))


if __name__ == "__main__":
    main()
