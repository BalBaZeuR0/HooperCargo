"""PPO training of the cargo hopper with a staged curriculum and live rlpanel logging.

    python -m hoopercargo.train --run-dir runs/main --steps-per-stage 50e6,100e6,150e6

Jump start (`--guide-schedule 1000,600,300,0`): each stage is split into equal
chunks; in chunk i the PD guide flies the first U{0..g_i} steps of every training
episode, so PPO sees landings from the start; the guide fades out over the chunks.
Evaluation always runs the policy alone.

Residual policy (`--residual 0.3`): the PD controller always flies and PPO learns
only a bounded correction on top of it, in training and in evaluation.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import functools
import json
from pathlib import Path
import time

from brax.envs import wrappers as brax_wrappers
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import flax
import jax
import jax.numpy as jp

from .config import HopperParams, parse_stage, stage_name
from .env import HooperCargoEnv

DEFAULT_PPO = dict(
    unroll_length=20,
    batch_size=512,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    discounting=0.995,   # ~4 s horizon: a 20 m flight must see the landing reward
    reward_scaling=0.1,
    normalize_observations=True,
    num_evals=20,
    num_eval_envs=128,
    deterministic_eval=True,
    policy_hidden_layer_sizes=(256, 256, 256),
    value_hidden_layer_sizes=(256, 256, 256),
)

# for tests and smoke runs
TINY_PPO = dict(
    episode_length=50, unroll_length=8, batch_size=8, num_minibatches=2, num_evals=2,
    num_eval_envs=4, policy_hidden_layer_sizes=(16,), value_hidden_layer_sizes=(16,),
)

NETWORK_KEYS = ("policy_hidden_layer_sizes", "value_hidden_layer_sizes")


class FreshAutoResetWrapper(brax_wrappers.training.Wrapper):
    """Auto-reset that draws a new random scenario for every episode.

    Brax's AutoResetWrapper restores the first reset state, so each training env
    would replay one pad / cargo / wind (and gust sequence) forever.
    """

    def reset(self, rng: jax.Array):
        state = self.env.reset(rng)
        state.info["reset_rng"] = jax.vmap(lambda k: jax.random.fold_in(k, 1))(rng)
        return state

    def step(self, state, action):
        if "steps" in state.info:
            state.info.update(steps=jp.where(state.done, 0, state.info["steps"]))
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)
        keys = jax.vmap(jax.random.split)(state.info["reset_rng"])
        fresh = self.env.reset(keys[:, 1])

        def where_done(new, old):
            done = state.done.reshape((-1,) + (1,) * (old.ndim - 1))
            return jp.where(done, new, old)

        pipeline_state = jax.tree.map(where_done, fresh.pipeline_state, state.pipeline_state)
        obs = where_done(fresh.obs, state.obs)
        state.info["reset_rng"] = keys[:, 0]
        return state.replace(pipeline_state=pipeline_state, obs=obs)


def wrap_fresh_reset(env, episode_length: int = 1000, action_repeat: int = 1,
                     randomization_fn=None):
    """Drop-in for `brax.envs.training.wrap` with fresh scenarios on auto-reset."""
    env = brax_wrappers.training.EpisodeWrapper(env, episode_length, action_repeat)
    env = brax_wrappers.training.VmapWrapper(env)
    return FreshAutoResetWrapper(env)


def ppo_config(scale: float, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULT_PPO, episode_length=HopperParams.for_scale(scale).episode_steps)
    cfg.update(overrides or {})
    return cfg


def make_networks_factory(cfg: dict, zero_mean_init: bool = False):
    """Brax PPO networks; `zero_mean_init` starts the policy mean at exactly 0.

    For the residual policy, a randomly initialised network adds a systematic,
    state-dependent correction (several degrees of gimbal) from the first evaluation,
    which pushed the PD off the pad (try9a: 97% -> 0% in one iteration). Zeroing the
    mean columns of the last layer (Silver et al. 2018) starts training at pure PD;
    the std columns stay random, so exploration is unchanged.
    """
    base = functools.partial(ppo_networks.make_ppo_networks,
                             policy_hidden_layer_sizes=tuple(cfg["policy_hidden_layer_sizes"]),
                             value_hidden_layer_sizes=tuple(cfg["value_hidden_layer_sizes"]))
    if not zero_mean_init:
        return base

    def factory(observation_size, action_size, **kw):
        nets = base(observation_size, action_size, **kw)
        policy = nets.policy_network
        last = f"hidden_{len(cfg['policy_hidden_layer_sizes'])}"

        def init(key):
            params = flax.core.unfreeze(policy.init(key))
            layer = params["params"][last]
            layer["kernel"] = layer["kernel"].at[:, :action_size].set(0.0)
            layer["bias"] = layer["bias"].at[:action_size].set(0.0)
            return params

        return nets.replace(policy_network=dataclasses.replace(policy, init=init))

    return factory


def train(run_dir: Path, steps_per_stage: list[int], num_envs: int, seed: int, scale: float,
          stages: list[int], randomize_payload: bool, ppo_overrides: dict | None = None,
          guide_schedule: list[int] | None = None, residual: float = 0.0) -> Path:
    from rlpanel import Panel

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = ppo_config(scale, ppo_overrides)
    guide_schedule = list(guide_schedule or [0])
    config = dict(steps_per_stage=list(steps_per_stage), num_envs=num_envs, seed=seed, scale=scale,
                  stages=list(stages), randomize_payload=randomize_payload,
                  guide_schedule=guide_schedule, residual=float(residual),
                  ppo={k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    total = int(sum(steps_per_stage))
    panel = Panel(project="HooperCargo", run=f"{run_dir.name}/seed_{seed}", seed=seed,
                  config=config, total_steps=total)
    train_kwargs = {k: v for k, v in cfg.items() if k not in NETWORK_KEYS}
    network_factory = make_networks_factory(cfg, zero_mean_init=residual > 0)

    csv_file = open(run_dir / "progress.csv", "w", newline="")
    writer = None
    params = None
    offset = 0
    last_success = float("nan")
    last_step = 0
    out = None
    try:
        chunks = [(stage, steps // len(guide_schedule), g)
                  for stage, steps in zip(stages, steps_per_stage) for g in guide_schedule]
        for i, (stage, steps, guide) in enumerate(chunks):
            env_kw = dict(stage=stage, scale=scale, randomize_payload=randomize_payload,
                          observe_payload=randomize_payload, residual=residual)
            env = HooperCargoEnv(guide_steps=guide, **env_kw)
            eval_env = HooperCargoEnv(**env_kw)
            t0 = time.time()

            def progress(step, metrics, stage=stage, offset=offset, t0=t0, guide=guide):
                nonlocal writer, last_success, last_step
                last_step = int(step)
                row = {k: float(v) for k, v in metrics.items() if jp.ndim(v) == 0}
                row.update(global_step=offset + int(step), stage=stage, guide=guide,
                           wall_seconds=time.time() - t0)
                fixed = ("global_step", "stage", "guide", "wall_seconds")
                if writer is None:
                    fields = list(fixed) + sorted(k for k in row if k not in fixed)
                    writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                last_success = row.get("eval/episode_success", last_success)
                # rlpanel's summary tiles read these standard names
                aliases = {"eval/mean_reward": "eval/episode_reward",
                           "eval/success_rate": "eval/episode_success",
                           "eval/mean_ep_length": "eval/avg_episode_length"}
                extra = {a: row[k] for a, k in aliases.items() if k in row}
                panel.log({**row, **extra, "stage": stage}, step=offset + int(step))
                panel.progress(offset + int(step), total)

            _, params, _ = ppo.train(
                environment=env, eval_env=eval_env, num_timesteps=int(steps), num_envs=num_envs,
                seed=seed + 100 * stage + i, network_factory=network_factory, progress_fn=progress,
                wrap_env_fn=wrap_fresh_reset,
                restore_params=params, restore_value_fn=True, **train_kwargs)
            out = run_dir / f"params_stage{stage_name(stage)}.pkl"
            brax_model.save_params(str(out), params)
            # Brax rounds the step budget up to whole batches: continue from what it ran
            offset += max(int(steps), last_step)
            last_step = 0
        panel.result({"final_eval_success": last_success})
    finally:
        csv_file.close()
        panel.finish()
    return out


def benchmark(num_envs: int, steps: int, scale: float) -> float:
    env = HooperCargoEnv(stage=2, scale=scale)
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    s = reset(jax.random.split(jax.random.PRNGKey(0), num_envs))
    a = jp.zeros((num_envs, 3))
    jax.block_until_ready(step(s, a))
    t0 = time.time()
    for _ in range(steps):
        s = step(s, a)
    jax.block_until_ready(s.obs)
    rate = num_envs * steps / (time.time() - t0)
    print(f"{jax.devices()[0].platform}: {rate:,.0f} env-steps/s ({num_envs} envs)")
    return rate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/main"))
    ap.add_argument("--steps-per-stage", default="50e6,100e6,150e6")
    ap.add_argument("--stages", default="0,1,2",
                    help="curriculum stages, e.g. H,0,1,2 (H = warm-up hop, logged as stage -1)")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-payload-randomization", action="store_true")
    ap.add_argument("--guide-schedule", default="0",
                    help="max PD-guided steps per chunk, e.g. 1000,600,300,0 (0 = no jump start)")
    ap.add_argument("--residual", type=float, default=0.0,
                    help="residual policy: action = PD + residual * policy (0 = policy flies alone)")
    ap.add_argument("--discount", type=float, default=None,
                    help="PPO discount (default 0.995); 0.999 sees a landing ~20 s ahead")
    ap.add_argument("--learning-rate", type=float, default=None, help="PPO learning rate (default 3e-4)")
    ap.add_argument("--benchmark", action="store_true", help="measure env-steps/s and exit")
    args = ap.parse_args(argv)
    if args.benchmark:
        benchmark(args.num_envs, 100, args.scale)
        return
    steps = [int(float(x)) for x in args.steps_per_stage.split(",")]
    stages = [parse_stage(x) for x in args.stages.split(",")]
    if len(steps) != len(stages):
        ap.error("--steps-per-stage and --stages need the same number of entries")
    guide = [int(float(x)) for x in args.guide_schedule.split(",")]
    overrides = {k: v for k, v in (("discounting", args.discount),
                                   ("learning_rate", args.learning_rate)) if v is not None}
    out = train(args.run_dir, steps, args.num_envs, args.seed, args.scale, stages,
                not args.no_payload_randomization, guide_schedule=guide,
                residual=args.residual, ppo_overrides=overrides)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
