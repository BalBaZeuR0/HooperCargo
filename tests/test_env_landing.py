import math

import jax
import jax.numpy as jnp
import pytest

from hoopercargo.env import HooperCargoEnv

KEY = jax.random.PRNGKey(3)


@pytest.fixture(scope="module")
def env():
    return HooperCargoEnv(stage=0)


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step)


def run_until_done(env, step, state, action_before, action_after=None, max_seconds=45.0):
    action_after = action_before if action_after is None else action_after
    for _ in range(int(max_seconds / env.dt)):
        a = action_after if bool(state.pipeline_state.touched) else action_before
        state = step(state, jnp.asarray(a, dtype=jnp.float32))
        if bool(state.done):
            return state
    raise AssertionError("episode never ended")


def drop(env, height, tilt_deg=0.0, offset_xy=(0.0, 0.0), vz=0.0, vxy=0.0):
    """Hopper above the landing pad, already airborne, engine idle."""
    s = jax.jit(env.reset)(KEY)
    ps = s.pipeline_state
    d = ps.data
    p = env.params
    xy = ps.pad_xy + jnp.asarray(offset_xy)
    half = math.radians(tilt_deg) / 2
    qpos = (d.qpos.at[0:2].set(xy).at[2].set(p.rest_height + height)
            .at[3:7].set(jnp.array([math.cos(half), math.sin(half), 0.0, 0.0])))
    qvel = jnp.zeros_like(d.qvel).at[0].set(vxy).at[2].set(vz)
    from mujoco import mjx
    d = mjx.forward(env.model_for(ps.cargo, ps.fuel), d.replace(qpos=qpos, qvel=qvel))
    return s.replace(pipeline_state=ps.replace(data=d, airborne=jnp.array(True)))


def m(state, key):
    return float(state.metrics[key])


IDLE, FULL = [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]


def test_upright_slow_drop_on_pad_is_success(env, step):
    s = run_until_done(env, step, drop(env, 0.1), IDLE)
    assert m(s, "success") == 1.0
    assert m(s, "landing_error") < 0.05
    assert float(s.reward) > 0


def test_tilted_drop_tips_over_is_fail(env, step):
    s = run_until_done(env, step, drop(env, 0.1, tilt_deg=50.0), IDLE)
    assert m(s, "fail_tilt") == 1.0 and m(s, "success") == 0.0
    assert float(s.reward) < 0


def test_off_pad_landing_is_fail(env, step):
    s = run_until_done(env, step, drop(env, 0.1, offset_xy=(3.0, 0.0)), IDLE)
    assert m(s, "fail_offpad") == 1.0 and m(s, "success") == 0.0


def test_hard_touchdown_is_fail(env, step):
    s = run_until_done(env, step, drop(env, 0.0, vz=-3.0), IDLE)
    assert m(s, "fail_hard") == 1.0 and m(s, "success") == 0.0
    assert m(s, "touch_vz") >= 2.0


def test_sideways_touchdown_is_fail(env, step):
    s = run_until_done(env, step, drop(env, 0.0, vz=-0.5, vxy=1.5), IDLE)
    assert m(s, "success") == 0.0


def test_launch_contact_is_not_touchdown(env, step):
    s = jax.jit(env.reset)(KEY)
    for _ in range(int(1.0 / env.dt)):
        s = step(s, jnp.asarray(IDLE))
    assert not bool(s.pipeline_state.touched) and not bool(s.pipeline_state.airborne)


def test_never_taking_off_times_out(env, step):
    s = run_until_done(env, step, jax.jit(env.reset)(KEY), IDLE)
    assert m(s, "fail_timeout") == 1.0 and m(s, "success") == 0.0
    assert int(s.pipeline_state.step) == env.params.episode_steps


def test_engine_cut_after_touchdown(env, step):
    s = run_until_done(env, step, drop(env, 0.1), IDLE, action_after=FULL)
    assert m(s, "success") == 1.0   # full throttle after touchdown did not lift it again


def test_metrics_only_on_final_step(env, step):
    s = step(drop(env, 0.1), jnp.asarray(IDLE))
    assert not bool(s.done)
    assert all(float(v) == 0.0 for v in s.metrics.values())
