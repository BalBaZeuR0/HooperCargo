"""The observation is egocentric: turning the whole scene about the vertical changes nothing."""
import jax
import jax.numpy as jnp
from mujoco import mjx

from hoopercargo.env import OBS_SIZE, HooperCargoEnv

KEY = jax.random.PRNGKey(0)


def rot_z(a):
    c, s = jnp.cos(a), jnp.sin(a)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def scene(env, base, yaw, pad_xy, vel):
    """`base` with the hopper at heading `yaw`, flying at world velocity `vel`, pad at `pad_xy`."""
    s = base.pipeline_state
    qpos = s.data.qpos.at[2].add(2.0).at[3:7].set(
        jnp.array([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)]))
    qvel = s.data.qvel.at[0:3].set(vel).at[3:6].set(jnp.array([0.1, -0.2, 0.3]))
    data = mjx.forward(env.model_for(s.cargo, s.fuel), s.data.replace(qpos=qpos, qvel=qvel))
    return env._observe(s.replace(data=data, pad_xy=pad_xy))


def test_observation_size_matches_layout():
    env = HooperCargoEnv(stage=0)
    s = jax.jit(env.reset)(KEY)
    assert s.obs.shape == (OBS_SIZE,) == (env.observation_size,)


def test_observation_is_invariant_to_heading():
    env = HooperCargoEnv(stage=0)
    base = jax.jit(env.reset)(KEY)
    pad, vel = jnp.array([8.0, 3.0]), jnp.array([1.0, -2.0, 0.5])
    a = scene(env, base, 0.0, pad, vel)
    for turn in (0.7, 2.0, -2.9):
        R = rot_z(turn)
        b = scene(env, base, turn, (R @ jnp.append(pad, 0.0))[0:2], R @ vel)
        assert jnp.allclose(a, b, atol=1e-4), turn


def test_pad_straight_ahead_is_on_the_body_x_axis():
    env = HooperCargoEnv(stage=0)
    base = jax.jit(env.reset)(KEY)
    yaw = 1.2
    obs = scene(env, base, yaw, 10.0 * jnp.array([jnp.cos(yaw), jnp.sin(yaw)]), jnp.zeros(3))
    rel_pad = obs[0:3]
    assert rel_pad[0] > 0.1 and abs(float(rel_pad[1])) < 1e-4 and rel_pad[2] < 0.0
