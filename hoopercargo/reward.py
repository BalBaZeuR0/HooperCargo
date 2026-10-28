"""Guidance velocity field and reward terms.

The per-step shaping is small and positive; summed over a whole episode it
stays below the success bonus, so the agent can neither farm reward by
hovering nor prefer an early crash to a long flight.

Progress toward the pad is rewarded per metre closed; it telescopes to the
start-to-finish distance change, so it cannot be farmed either. A soft,
upright landing just off the pad earns part of the fail penalty back, which
gives the policy a gradient toward the pad before it ever finds a success.
Without these two terms every failure costs the same, and the discounted
agent learnt to stay airborne and drift away to postpone the penalty.
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
    near_miss=100.0,   # refund for a soft upright landing next to the pad
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
    return (REWARD["track"] * jp.exp(-jp.sum(err**2) / REWARD["track_width"])
            + REWARD["upright"] * jp.exp(-tilt**2 / REWARD["tilt_width"])
            - REWARD["throttle"] * throttle
            - REWARD["smooth"] * jp.sum(d_action**2))


def event_reward(success: jax.Array, fail: jax.Array) -> jax.Array:
    return jp.where(success, REWARD["success"], jp.where(fail, REWARD["fail"], 0.0))


def progress_reward(dist_prev: jax.Array, dist: jax.Array, params: HopperParams) -> jax.Array:
    """Horizontal distance closed toward the pad this step (m, at scale 1)."""
    return REWARD["progress"] * (dist_prev - dist) / params.scale


def near_miss_bonus(landing_error: jax.Array, params: HopperParams) -> jax.Array:
    """Part of the fail penalty refunded for landing close to the pad."""
    return REWARD["near_miss"] * jp.exp(-(landing_error / (2.0 * params.pad_radius)) ** 2)
