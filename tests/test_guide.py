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


def test_residual_policy_adds_a_bounded_correction_to_pd():
    env = HooperCargoEnv(stage=0, residual=0.3)
    s = jax.jit(env.reset)(jax.random.PRNGKey(2))
    step = jax.jit(env.step)
    zero = step(s, jnp.zeros(3)).pipeline_state.last_action      # pure PD
    push = step(s, jnp.array([1.0, -1.0, 5.0])).pipeline_state.last_action
    expected = jnp.clip(zero + 0.3 * jnp.array([1.0, -1.0, 1.0]), -1.0, 1.0)
    assert float(zero[0]) > -1.0                                  # PD throttles up
    assert jnp.allclose(push, expected, atol=1e-5)


def test_residual_correction_costs_reward_in_flight():
    from hoopercargo.reward import REWARD
    env = HooperCargoEnv(stage=0, residual=0.3)
    s = jax.jit(env.reset)(jax.random.PRNGKey(4))
    step = jax.jit(env.step)
    prev = jax.lax.fori_loop(0, 100, lambda i, st: step(st, jnp.zeros(3)), s)   # airborne, PD only
    nxt = step(prev, jnp.zeros(3))
    landing = jax.jit(env._landing)
    ps, ns = prev.pipeline_state, nxt.pipeline_state
    _, r_free, _, _ = landing(ps, ns, jnp.array(0.5), jnp.array(0.0))
    _, r_full, _, _ = landing(ps, ns, jnp.array(0.5), jnp.array(1.0))
    assert not bool(ns.touched)
    assert float(r_free - r_full) == pytest.approx(REWARD["correction"], abs=1e-6)


def test_residual_policy_starts_at_pure_pd():
    from brax.training.agents.ppo import networks as ppo_networks
    from hoopercargo.train import TINY_PPO, make_networks_factory, ppo_config
    cfg = ppo_config(1.0, TINY_PPO)
    nets = make_networks_factory(cfg, zero_mean_init=True)(11, 3)
    params = nets.policy_network.init(jax.random.PRNGKey(0))
    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 11)) * 10.0
    out = nets.policy_network.apply(None, params, obs)
    assert out.shape == (5, 6)
    assert jnp.allclose(out[:, :3], 0.0)                 # mean = 0 -> correction 0 -> PD
    assert float(jnp.abs(out[:, 3:]).sum()) > 0.0       # std head still random
    act = ppo_networks.make_inference_fn(nets)((None, params), deterministic=True)
    a, _ = act(obs[0], jax.random.PRNGKey(2))
    assert jnp.allclose(a, 0.0)
