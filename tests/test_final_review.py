"""Regression tests for the whole-branch review findings."""
import jax
import jax.numpy as jnp
import pytest

from hoopercargo.env import HooperCargoEnv
from hoopercargo.payload import Cargo
from hoopercargo.train import TINY_PPO, ppo_config, train, wrap_fresh_reset

KEY = jax.random.PRNGKey(0)
MASS_OBS = 15   # rel_pad 3 + vel 3 + up 3 + omega 3 + tvc 2 + fuel 1


def test_unobserved_payload_gives_constant_mass_observation():
    env = HooperCargoEnv(stage=2, observe_payload=False)
    s = jax.jit(jax.vmap(env.reset))(jax.random.split(KEY, 64))
    assert jnp.allclose(s.obs[:, MASS_OBS], 0.5)
    observed = HooperCargoEnv(stage=2)
    s = jax.jit(jax.vmap(observed.reset))(jax.random.split(KEY, 64))
    assert float(jnp.std(s.obs[:, MASS_OBS])) > 0.1


def test_auto_reset_draws_fresh_scenarios():
    env = HooperCargoEnv(stage=0)
    wrapped = wrap_fresh_reset(env, episode_length=2, action_repeat=1)
    s = jax.jit(wrapped.reset)(jax.random.split(KEY, 4))
    first_pad = s.pipeline_state.pad_xy
    step = jax.jit(wrapped.step)
    for _ in range(2):            # episode_length = 2 -> auto reset after step 2
        s = step(s, jnp.zeros((4, 3)))
    assert not jnp.allclose(s.pipeline_state.pad_xy, first_pad)
    assert jnp.isfinite(s.obs).all()


def test_brax_default_auto_reset_replays_the_same_scenario():
    # the review finding this wrapper fixes
    from brax.envs.wrappers.training import wrap
    env = HooperCargoEnv(stage=0)
    wrapped = wrap(env, episode_length=2, action_repeat=1)
    s = jax.jit(wrapped.reset)(jax.random.split(KEY, 4))
    first_pad = s.pipeline_state.pad_xy
    step = jax.jit(wrapped.step)
    for _ in range(2):
        s = step(s, jnp.zeros((4, 3)))
    assert jnp.allclose(s.pipeline_state.pad_xy, first_pad)


def test_wind_tilts_the_vehicle_through_centre_of_pressure():
    env = HooperCargoEnv(stage=0)
    s = jax.jit(env.reset)(KEY)
    d = s.pipeline_state.data
    # upright, at rest, 6 m/s wind along +x: drag (1.59 N) acts 0.15 m above the CoM,
    # so it pitches the vehicle about world +y by ~0.24 N m
    _, torque = env.aero_wrench(d, jnp.array([6.0, 0.0, 0.0]))
    assert 0.2 < float(torque[1]) < 0.3 and abs(float(torque[0])) < 1e-6


def test_eval_uses_deterministic_policy():
    assert ppo_config(1.0)["deterministic_eval"] is True


@pytest.fixture(scope="module")
def fixed_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("fixed_run")
    return train(run, steps_per_stage=[1024], num_envs=8, seed=0, scale=1.0, stages=[0],
                 randomize_payload=False, ppo_overrides=TINY_PPO)


def test_eval_env_follows_run_config(fixed_run):
    from hoopercargo.evaluate import make_eval_env
    env = make_eval_env(fixed_run, stage=2)
    assert env.params.scale == 1.0
    assert env.observe_payload is False          # trained blind to payload
    assert env.randomize_payload is True         # but evaluated on random payloads


def test_eval_rejects_scale_mismatch(fixed_run, tmp_path):
    from hoopercargo.evaluate import evaluate
    with pytest.raises(ValueError, match="scale"):
        evaluate(fixed_run, 2, 0, 2, 2.0, tmp_path / "x.csv")
