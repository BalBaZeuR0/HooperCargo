import jax.numpy as jnp

from hoopercargo.config import HopperParams
import math

from hoopercargo.reward import (REWARD, TAKEOFF_HEIGHT, braking_distance, event_reward,
                                guidance_velocity, lean_bonus, near_miss_bonus, progress_reward,
                                step_reward, tilt_cost, tilt_limit)

P = HopperParams.for_scale(1.0)


def test_guidance_points_to_pad_and_climbs_when_far():
    v = guidance_velocity(jnp.array([50.0, 0.0]), jnp.array(5.0), P)
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
    tilted = step_reward(vd, vd, jnp.array(1.0), jnp.array(0.5), jnp.zeros(3), P,
                         limit=math.radians(45.0))
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


def test_takeoff_gate_climbs_first_then_opens_horizontal_speed():
    on_ground = guidance_velocity(jnp.array([30.0, 0.0]), jnp.array(0.0), P)
    half = guidance_velocity(jnp.array([30.0, 0.0]), jnp.array(0.5 * TAKEOFF_HEIGHT), P)
    up = guidance_velocity(jnp.array([30.0, 0.0]), jnp.array(TAKEOFF_HEIGHT), P)
    assert float(on_ground[0]) == 0.0 and float(on_ground[2]) > 0
    assert 0.0 < float(half[0]) < float(up[0])
    # short hops still cruise above the gate, so it fully opens
    short = guidance_velocity(jnp.array([5.0, 0.0]), jnp.array(TAKEOFF_HEIGHT + 0.5), P)
    assert float(short[2]) > 0


def test_tilt_bands_free_to_45_light_to_50_then_growing():
    lim = math.radians(45.0)
    c = lambda deg: float(tilt_cost(jnp.array(math.radians(deg)), lim))
    assert c(0) == 0.0 and c(30) == 0.0 and c(45) == 0.0
    assert 0.0 < c(47) < c(50) <= 0.25 + 1e-6
    assert c(65) > c(50) and c(80) > c(65)


def test_tilt_limit_narrows_to_15_deg_over_the_pad():
    v = jnp.array([6.0, 0.0])
    far = float(tilt_limit(jnp.array(80.0), v, P))
    over = float(tilt_limit(jnp.array(0.5), v, P))
    assert abs(far - math.radians(45.0)) < 1e-6
    assert abs(over - math.radians(15.0)) < 1e-6
    mid = float(tilt_limit(P.pad_radius + 0.5 * braking_distance(v, P), v, P))
    assert math.radians(15.0) < mid < math.radians(45.0)


def test_lean_bonus_only_when_leaning_and_moving_toward_pad_in_cruise():
    v = jnp.array([3.0, 0.0])
    lean = jnp.array(math.radians(15.0))
    args = dict(d=jnp.array(50.0), height=jnp.array(5.0), v_xy=v, params=P)
    full = float(lean_bonus(lean, jnp.array(3.0), **args))
    assert abs(full - REWARD["lean"]) < 1e-6
    assert float(lean_bonus(lean, jnp.array(0.0), **args)) == 0.0          # hovering: nothing
    assert float(lean_bonus(-lean, jnp.array(3.0), **args)) == 0.0         # leaning away: nothing
    assert float(lean_bonus(2 * lean, jnp.array(3.0), **args)) == full     # saturates at 15 deg
    near = dict(args, d=jnp.array(2.0))
    assert float(lean_bonus(lean, jnp.array(3.0), **near)) == 0.0          # braking zone: nothing
    low = dict(args, height=jnp.array(1.0))
    assert float(lean_bonus(lean, jnp.array(3.0), **low)) == 0.0           # still taking off
    # a whole episode of it is worth less than a landing
    assert REWARD["lean"] * P.episode_steps < 0.25 * REWARD["success"]
