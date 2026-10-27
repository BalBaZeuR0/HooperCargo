"""Horizontal wind: random mean wind plus Ornstein-Uhlenbeck gusts, and drag."""
from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jp

from .config import AIR_DENSITY, HopperParams, StageConfig

GUST_TAU = 2.0   # s, gust correlation time


@struct.dataclass
class WindState:
    mean: jax.Array   # [3] m/s, z = 0
    gust: jax.Array   # [3] m/s, z = 0


def sample_wind(key: jax.Array, stage: StageConfig) -> WindState:
    k_s, k_a = jax.random.split(key)
    speed = jax.random.uniform(k_s, (), minval=0.0, maxval=stage.wind_max)
    a = jax.random.uniform(k_a, (), minval=0.0, maxval=2.0 * jp.pi)
    return WindState(mean=jp.array([speed * jp.cos(a), speed * jp.sin(a), 0.0]), gust=jp.zeros(3))


def step_gust(key: jax.Array, gust: jax.Array, sigma: jax.Array, dt: float,
              tau: float = GUST_TAU) -> jax.Array:
    """Exact OU discretisation; stationary std = sigma on x and y."""
    decay = jp.exp(-dt / tau)
    noise = jax.random.normal(key, (3,)) * jp.array([1.0, 1.0, 0.0])
    return gust * decay + sigma * jp.sqrt(1.0 - decay**2) * noise


def drag_force(v_body: jax.Array, wind_vel: jax.Array, params: HopperParams) -> jax.Array:
    v_rel = wind_vel - v_body
    return 0.5 * AIR_DENSITY * params.drag_cd * params.drag_area * jp.linalg.norm(v_rel) * v_rel
