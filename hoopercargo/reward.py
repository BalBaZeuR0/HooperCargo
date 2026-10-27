"""Guidance velocity field and reward terms.

The per-step shaping is small and positive; summed over a whole episode it
stays below the success bonus, so the agent can neither farm reward by
hovering nor prefer an early crash to a long flight.
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
    track_width=4.0,   # (m/s)^2
    tilt_width=0.1,    # rad^2
    success=200.0,
    fail=-100.0,
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
