import jax
import jax.numpy as jnp
import pytest

from hoopercargo.env import HooperCargoEnv
from hoopercargo.payload import Cargo

KEY = jax.random.PRNGKey(0)
HOPPER = HooperCargoEnv(stage=0).ids.hopper


@pytest.fixture(scope="module")
def env0():
    return HooperCargoEnv(stage=0)


@pytest.fixture(scope="module")
def step0(env0):
    return jax.jit(env0.step)


def rollout(env, step, action, seconds, state=None):
    state = jax.jit(env.reset)(KEY) if state is None else state
    a = jnp.asarray(action, dtype=jnp.float32)
    for _ in range(int(round(seconds / env.dt))):
        state = step(state, a)
    return state


def with_cargo(env, mass):
    s = jax.jit(env.reset)(KEY)
    ps = s.pipeline_state
    return s.replace(pipeline_state=ps.replace(
        cargo=Cargo(mass=jnp.array(mass), offset=jnp.array([0.0, 0.0, 0.175]))))


def com_z(state, env=None):
    return float(state.pipeline_state.data.subtree_com[HOPPER, 2])


def test_reset_shapes_and_jit_vmap():
    env = HooperCargoEnv(stage=2)
    s = jax.jit(jax.vmap(env.reset))(jax.random.split(KEY, 8))
    assert s.obs.shape == (8, 25)
    assert env.observation_size == 25 and env.action_size == 3
    s2 = jax.jit(jax.vmap(env.step))(s, jnp.zeros((8, 3)))
    assert jnp.isfinite(s2.obs).all()


def test_zero_throttle_stays_on_ground(env0, step0):
    start = jax.jit(env0.reset)(KEY)
    s = rollout(env0, step0, [-1, 0, 0], 1.0, start)
    assert com_z(s) < com_z(start) + 0.01
    assert jnp.isfinite(s.obs).all()


def test_full_throttle_climbs_and_heavier_climbs_slower(env0, step0):
    start = jax.jit(env0.reset)(KEY)
    light = rollout(env0, step0, [1, 0, 0], 1.0, with_cargo(env0, 0.0))
    heavy = rollout(env0, step0, [1, 0, 0], 1.0, with_cargo(env0, 0.8))
    assert com_z(light) > com_z(heavy) > com_z(start) + 0.5


def test_fuel_burn_rate_matches_isp(env0, step0):
    start = jax.jit(env0.reset)(KEY)
    s = rollout(env0, step0, [1, 0, 0], 2.0, start)
    burned = float(start.pipeline_state.fuel - s.pipeline_state.fuel)
    assert abs(burned - 45.0 / (150.0 * 9.80665) * 2.0) < 0.002


def test_empty_tank_gives_no_thrust(env0, step0):
    start = jax.jit(env0.reset)(KEY)
    empty = start.replace(pipeline_state=start.pipeline_state.replace(fuel=jnp.array(0.0)))
    s = rollout(env0, step0, [1, 0, 0], 1.0, empty)
    # body origin, not CoM: removing the fuel itself raises the CoM by ~2 cm
    assert float(s.pipeline_state.data.qpos[2]) < float(start.pipeline_state.data.qpos[2]) + 0.01
    assert float(s.pipeline_state.fuel) == 0.0


def test_tank_runs_dry_without_going_negative(env0, step0):
    start = jax.jit(env0.reset)(KEY)
    low = start.replace(pipeline_state=start.pipeline_state.replace(fuel=jnp.array(0.02)))
    s = rollout(env0, step0, [1, 0, 0], 1.0, low)   # 0.02 kg lasts ~0.65 s at 45 N
    assert float(s.pipeline_state.fuel) == 0.0
    assert jnp.isfinite(s.obs).all()


def test_uncontrolled_flight_stays_finite(env0, step0):
    # open loop the hopper tumbles and crashes; stepping on must stay finite
    s = rollout(env0, step0, [1, 0, 0], 25.0)
    assert jnp.isfinite(s.obs).all() and float(s.pipeline_state.fuel) >= 0.0


def test_pad_distance_in_stage_range():
    env = HooperCargoEnv(stage=2)
    s = jax.jit(jax.vmap(env.reset))(jax.random.split(KEY, 1024))
    d = jnp.linalg.norm(s.pipeline_state.pad_xy, axis=1)
    assert (d >= 20).all() and (d <= 100).all()


def test_tvc_tilts_the_vehicle(env0, step0):
    s = rollout(env0, step0, [1, 0, 0], 0.5)
    s = rollout(env0, step0, [0.2, 0.0, 0.5], 1.0, s)
    up = s.pipeline_state.data.xmat[HOPPER][:, 2]
    assert float(up[2]) < 0.99   # the gimbal produces a real torque
