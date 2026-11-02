import jax
import jax.numpy as jnp
import pytest

from hoopercargo.env import HooperCargoEnv


@pytest.fixture(scope="module")
def guided():
    env = HooperCargoEnv(stage=0, guide_steps=10_000)
    return env, jax.jit(env.reset), jax.jit(env.step)


def test_guide_overrides_policy_then_hands_over(guided):
    env, reset, step = guided
    s = reset(jax.random.PRNGKey(3))
    left = int(s.pipeline_state.guide_left)
    assert 0 <= left <= 10_000
    junk = jnp.array([-1.0, 1.0, 1.0])            # engine off, full gimbal
    s1 = step(s, junk)
    if left > 0:
        assert float(s1.pipeline_state.last_action[0]) > -1.0   # the guide throttled up
        assert int(s1.pipeline_state.guide_left) == left - 1


def test_unguided_env_obeys_the_policy():
    env = HooperCargoEnv(stage=0)
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert int(s.pipeline_state.guide_left) == 0
    a = jnp.array([-1.0, 0.5, -0.5])
    s1 = jax.jit(env.step)(s, a)
    assert jnp.allclose(s1.pipeline_state.last_action, a)


def test_guide_takes_off_and_reports_it(guided):
    env, reset, step = guided
    s = reset(jax.random.PRNGKey(1))
    s = s.replace(pipeline_state=s.pipeline_state.replace(guide_left=jnp.array(10_000)))
    s = jax.lax.fori_loop(0, 150, lambda i, st: step(st, jnp.zeros(3)), s)
    assert bool(s.pipeline_state.airborne)
