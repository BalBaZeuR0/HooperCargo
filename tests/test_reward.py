import jax.numpy as jnp

from hoopercargo.config import HopperParams
from hoopercargo.reward import (REWARD, event_reward, guidance_velocity, near_miss_bonus,
                                progress_reward, step_reward)

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


def test_step_shaping_is_never_positive_so_hovering_cannot_be_farmed():
    vd = jnp.zeros(3)
    best = step_reward(vd, vd, jnp.array(0.0), jnp.array(0.0), jnp.zeros(3), P)
    assert float(best) == 0.0
    assert float(step_reward(vd + 1.0, vd, jnp.array(0.3), jnp.array(0.6), jnp.ones(3), P)) < 0.0
    # to the discounted agent, even the worst shaping stream costs far less than a crash
    from hoopercargo.train import DEFAULT_PPO
    worst = REWARD["track"] + REWARD["upright"] + REWARD["throttle"]
    assert worst / (1.0 - DEFAULT_PPO["discounting"]) < -0.25 * REWARD["fail"]


def test_event_rewards():
    assert event_reward(jnp.array(True), jnp.array(False)) == REWARD["success"]
    assert event_reward(jnp.array(False), jnp.array(True)) == REWARD["fail"]
    assert event_reward(jnp.array(False), jnp.array(False)) == 0.0


def test_progress_rewards_closing_in_and_telescopes():
    assert progress_reward(jnp.array(10.0), jnp.array(9.0), P) > 0
    assert progress_reward(jnp.array(10.0), jnp.array(11.0), P) < 0
    dists = jnp.array([20.0, 18.0, 19.0, 5.0, 0.5])
    total = sum(float(progress_reward(a, b, P)) for a, b in zip(dists[:-1], dists[1:]))
    assert abs(total - REWARD["progress"] * (20.0 - 0.5)) < 1e-4
    # even the longest stage-2 flight cannot out-earn a successful landing
    assert REWARD["progress"] * 100.0 < REWARD["success"]


def test_soft_offpad_landing_pays_less_than_success_but_more_than_failing():
    near = float(near_miss_bonus(jnp.array(1.6), P))
    far = float(near_miss_bonus(jnp.array(10.0), P))
    assert REWARD["success"] > near > far >= 0.0 > REWARD["fail"]
