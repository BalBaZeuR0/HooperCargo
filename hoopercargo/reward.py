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
"""
from __future__ import annotations

import jax
import jax.numpy as jp

from .config import HopperParams

REWARD = dict(
    track=0.05,        # velocity tracking, per step
    upright=0.025,     # staying upright, per step
    throttle=0.01,     # fuel use, per step
    smooth=0.01,       # action changes, per step
    progress=1.0,      # per metre of horizontal distance closed toward the pad
    track_width=4.0,   # (m/s)^2
    tilt_width=0.1,    # rad^2
    success=200.0,
    fail=-100.0,
    near_miss=200.0,   # soft upright landing off the pad, times closeness (< success)
)


def guidance_velocity(rel_xy: jax.Array, height: jax.Array, params: HopperParams) -> jax.Array:
    """Desired velocity: fly at a cruise altitude toward the pad, then descend onto it."""
    L, s = params.scale, jp.sqrt(params.scale)
    d = jp.linalg.norm(rel_xy)
    direction = rel_xy / jp.maximum(d, 1e-6)
    v_xy = direction * jp.minimum(8.0 * s, 0.5 * d)
    h_des = jp.clip(0.3 * d, 0.0, 15.0 * L)
    climb = jp.clip(0.8 * (h_des - height), -2.0 * s, 3.0 * s)
    descend = -jp.clip(0.5 * height + 0.3 * s, 0.3 * s, 0.75 * params.touchdown_vz_max)
    v_z = jp.where(d > params.pad_radius, climb, descend)
    return jp.array([v_xy[0], v_xy[1], v_z])


def step_reward(v: jax.Array, v_des: jax.Array, tilt: jax.Array, throttle: jax.Array,
                d_action: jax.Array, params: HopperParams) -> jax.Array:
    err = (v - v_des) / jp.sqrt(params.scale)
    return (-REWARD["track"] * (1.0 - jp.exp(-jp.sum(err**2) / REWARD["track_width"]))
            - REWARD["upright"] * (1.0 - jp.exp(-tilt**2 / REWARD["tilt_width"]))
            - REWARD["throttle"] * throttle
            - REWARD["smooth"] * jp.sum(d_action**2))


def event_reward(success: jax.Array, fail: jax.Array) -> jax.Array:
    return jp.where(success, REWARD["success"], jp.where(fail, REWARD["fail"], 0.0))


def progress_reward(dist_prev: jax.Array, dist: jax.Array, params: HopperParams) -> jax.Array:
    """Horizontal distance closed toward the pad this step (m, at scale 1)."""
    return REWARD["progress"] * (dist_prev - dist) / params.scale


def near_miss_bonus(landing_error: jax.Array, params: HopperParams) -> jax.Array:
    """Reward for a soft upright landing off the pad: the closer, the more."""
    return REWARD["near_miss"] * jp.exp(-(landing_error / (2.0 * params.pad_radius)) ** 2)
