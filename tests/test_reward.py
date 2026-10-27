import jax.numpy as jnp

from hoopercargo.config import HopperParams
from hoopercargo.reward import REWARD, event_reward, guidance_velocity, step_reward

P = HopperParams.for_scale(1.0)


def test_guidance_points_to_pad_and_climbs_when_far():
    v = guidance_velocity(jnp.array([50.0, 0.0]), jnp.array(0.0), P)
    assert v[0] > 0 and abs(float(v[1])) < 1e-6 and v[2] > 0
    assert float(v[0]) <= 8.0 + 1e-6


def test_guidance_descends_over_pad():
    v = guidance_velocity(jnp.array([0.2, 0.0]), jnp.array(5.0), P)
    assert v[2] < 0
    # gentle near the ground, below the touchdown limit
    v = guidance_velocity(jnp.array([0.0, 0.0]), jnp.array(0.1), P)
    assert -P.touchdown_vz_max < float(v[2]) < 0


def test_tracking_reward_higher_when_matching():
    vd = jnp.array([3.0, 0.0, 1.0])
    good = step_reward(vd, vd, jnp.array(0.0), jnp.array(0.5), jnp.zeros(3), P)
    bad = step_reward(vd + 2.0, vd, jnp.array(0.0), jnp.array(0.5), jnp.zeros(3), P)
    tilted = step_reward(vd, vd, jnp.array(0.5), jnp.array(0.5), jnp.zeros(3), P)
    assert good > bad and good > tilted


def test_whole_flight_shaping_is_smaller_than_success_bonus():
    vd = jnp.zeros(3)
    best = step_reward(vd, vd, jnp.array(0.0), jnp.array(0.0), jnp.zeros(3), P)
    assert float(best) * P.episode_steps < REWARD["success"]


def test_event_rewards():
    assert event_reward(jnp.array(True), jnp.array(False)) == REWARD["success"]
    assert event_reward(jnp.array(False), jnp.array(True)) == REWARD["fail"]
    assert event_reward(jnp.array(False), jnp.array(False)) == 0.0
