"""Guidance velocity field and reward terms.

The per-step shaping is never positive: tracking the guidance velocity and
staying upright only reduce a small per-step cost. A positive per-step term,
however small, is worth more to a discounted agent than a landing many
seconds away, and the policy learnt to hover until timeout to collect it.

Progress toward the pad is rewarded per metre closed; it telescopes to the
start-to-finish distance change, so it cannot be farmed either. A soft,
upright landing off the pad is not a failure in reward terms: it earns a
share of the success bonus that shrinks with the distance to the pad. Any
touchdown the vehicle survives therefore beats crashing, timing out or
leaving the area, and landing closer always pays more. With a flat penalty
for every failure, the discounted agent learnt to never touch down and to
drift away instead, postponing a penalty it could not avoid.

Tilt (angle from the vertical) is free up to 20 deg in cruise, enough for 8 m/s
into a 6 m/s headwind; it costs a little from 20 to 30 deg and more beyond. With
45 deg free (try6) the only attitude signal left was the 80 deg crash, too late
for a ~4 s discount horizon, and the policy tipped over. Near the pad the free
limit narrows to the 15 deg landing limit. The one
positive per-step term is a small lean bonus: leaning toward the pad, while
already moving toward it and outside the braking distance, earns up to `lean`
per step (saturating at 15 deg). It is zero when hovering, so it cannot be
farmed, and a whole episode of it stays well below the success bonus.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jp

from .config import G0, HopperParams

REWARD = dict(
    track=0.05,        # velocity tracking, per step
    upright=0.05,      # tilt beyond the free limit, per step
    throttle=0.01,     # fuel use, per step
    smooth=0.01,       # action changes, per step
    progress=1.0,      # per metre of horizontal distance closed toward the pad
    track_width=4.0,   # (m/s)^2
    success=200.0,
    fail=-100.0,
    near_miss=200.0,   # soft upright landing off the pad, times closeness (< success)
    lean=0.02,         # per step, leaning toward the pad in cruise (positive, capped)
    correction=0.02,   # residual policy: per step, times the mean squared correction
    early_end=0.05,    # per step left in the episode, when it ends by tipping over/leaving/not lifting
)

TAKEOFF_HEIGHT = 2.0                      # m (x L): horizontal guidance opens up to here
LEAN_TARGET = math.radians(15.0)          # lean bonus saturates here
TILT_FREE = math.radians(20.0)            # cruise: no tilt cost below this
TILT_WARN = math.radians(30.0)            # 20-30 deg: light warning cost
TILT_PAD = math.radians(15.0)             # free limit over the pad (= landing limit)
BRAKE_TILT = math.radians(15.0)           # tilt assumed for the braking distance


def guidance_velocity(rel_xy: jax.Array, height: jax.Array, params: HopperParams) -> jax.Array:
    """Desired velocity: climb, fly at a cruise altitude toward the pad, then descend onto it.

    Horizontal speed opens up linearly while climbing to `TAKEOFF_HEIGHT`, so the vehicle
    is not asked to bank next to the ground; the cruise altitude stays 1 m above that.
    """
    L, s = params.scale, jp.sqrt(params.scale)
    h_min = TAKEOFF_HEIGHT * L
    d = jp.linalg.norm(rel_xy)
    direction = rel_xy / jp.maximum(d, 1e-6)
    gate = jp.clip(height / h_min, 0.0, 1.0)
    v_xy = direction * jp.minimum(8.0 * s, 0.5 * d) * gate
    h_des = jp.clip(0.3 * d, h_min + 1.0 * L, 15.0 * L)
    climb = jp.clip(0.8 * (h_des - height), -2.0 * s, 3.0 * s)
    descend = -jp.clip(0.5 * height + 0.3 * s, 0.3 * s, 0.75 * params.touchdown_vz_max)
    v_z = jp.where(d > params.pad_radius, climb, descend)
    return jp.array([v_xy[0], v_xy[1], v_z])


def braking_distance(v_xy: jax.Array, params: HopperParams) -> jax.Array:
    """Horizontal distance to stop from |v_xy| at a 15 deg lean, plus the pad radius."""
    return jp.sum(v_xy**2) / (2.0 * G0 * math.tan(BRAKE_TILT)) + params.pad_radius


def tilt_limit(d: jax.Array, v_xy: jax.Array, params: HopperParams) -> jax.Array:
    """Free tilt: 20 deg in cruise, narrowing to 15 deg over the pad inside the braking distance."""
    far = jp.clip((d - params.pad_radius) / jp.maximum(braking_distance(v_xy, params), 1e-6),
                  0.0, 1.0)
    return TILT_PAD + (TILT_FREE - TILT_PAD) * far


def tilt_cost(tilt: jax.Array, limit: jax.Array) -> jax.Array:
    """0 below `limit`; a light ramp over the next 10 deg; growing beyond (1 per 30 deg)."""
    warn = jp.clip((tilt - limit) / (TILT_WARN - TILT_FREE), 0.0, 1.0)
    over = jp.maximum(tilt - limit - (TILT_WARN - TILT_FREE), 0.0) / math.radians(30.0)
    return 0.25 * warn + over


def lean_bonus(lean_toward: jax.Array, v_toward: jax.Array, d: jax.Array, height: jax.Array,
               v_xy: jax.Array, params: HopperParams) -> jax.Array:
    """Small reward for leaning toward the pad in cruise; zero unless moving toward it."""
    s = jp.sqrt(params.scale)
    cruising = (height >= TAKEOFF_HEIGHT * params.scale) & (d > braking_distance(v_xy, params))
    lean = jp.clip(lean_toward / LEAN_TARGET, 0.0, 1.0)
    moving = jp.clip(v_toward / (2.0 * s), 0.0, 1.0)
    return jp.where(cruising, REWARD["lean"] * lean * moving, 0.0)


def step_reward(v: jax.Array, v_des: jax.Array, tilt: jax.Array, throttle: jax.Array,
                d_action: jax.Array, params: HopperParams,
                limit: jax.Array | float = TILT_PAD) -> jax.Array:
    """Per-step cost (never positive). `limit` is the free tilt from `tilt_limit`."""
    err = (v - v_des) / jp.sqrt(params.scale)
    return (-REWARD["track"] * (1.0 - jp.exp(-jp.sum(err**2) / REWARD["track_width"]))
            - REWARD["upright"] * tilt_cost(tilt, limit)
            - REWARD["throttle"] * throttle
            - REWARD["smooth"] * jp.sum(d_action**2))


def event_reward(success: jax.Array, fail: jax.Array) -> jax.Array:
    return jp.where(success, REWARD["success"], jp.where(fail, REWARD["fail"], 0.0))


def early_end_penalty(early: jax.Array, steps_left: jax.Array) -> jax.Array:
    """Charge the steps an episode skips by ending in a crash.

    Every failure pays the same `fail`, and the per-step shaping is a cost, so with
    a long horizon (discount 0.999) the cheapest failure is the quickest: try10b
    tipped itself over 0.4 s after lift-off, every episode. Charging each skipped
    step at about the per-step cost of flying makes no crash cheaper than flying on.
    Touchdowns that settle without tipping over are exempt, so trying to land never
    costs more than hovering to the timeout.
    """
    return jp.where(early, -REWARD["early_end"] * steps_left, 0.0)


def progress_reward(dist_prev: jax.Array, dist: jax.Array, params: HopperParams) -> jax.Array:
    """Horizontal distance closed toward the pad this step (m, at scale 1)."""
    return REWARD["progress"] * (dist_prev - dist) / params.scale


def near_miss_bonus(landing_error: jax.Array, params: HopperParams) -> jax.Array:
    """Reward for a soft upright landing off the pad: the closer, the more."""
    return REWARD["near_miss"] * jp.exp(-(landing_error / (2.0 * params.pad_radius)) ** 2)
