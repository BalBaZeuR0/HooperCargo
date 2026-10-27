"""Cargo sampling and the mass properties of the payload and fuel bodies.

Payload and fuel are weld (joint-less) children of the hopper body.  Their
mass, CoM and inertia are written into the MJX model every control step, so
the simulator sees the true total mass, centre of mass and inertia.
"""
from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jp

from .config import HopperParams, StageConfig
from .model import BodyIds

MIN_MASS = 1e-4


@struct.dataclass
class Cargo:
    mass: jax.Array      # kg
    offset: jax.Array    # [3] CoM in the hopper body frame (m)


def sample_cargo(key: jax.Array, params: HopperParams, stage: StageConfig) -> Cargo:
    k_mass, k_z, k_r, k_a = jax.random.split(key, 4)
    mass = jax.random.uniform(k_mass, (), minval=0.0, maxval=stage.payload_max)
    z = jax.random.uniform(k_z, (), minval=params.bay_z_min, maxval=params.bay_z_max)
    # uniform over a disk: r = R * sqrt(u)
    r = params.bay_lateral_max * jp.sqrt(jax.random.uniform(k_r, ()))
    a = jax.random.uniform(k_a, (), minval=0.0, maxval=2.0 * jp.pi)
    return Cargo(mass=mass, offset=jp.array([r * jp.cos(a), r * jp.sin(a), z]))


def point_inertia(mass: jax.Array, radius: float) -> jax.Array:
    """Diagonal inertia of a solid sphere."""
    i = 0.4 * jp.maximum(mass, MIN_MASS) * radius**2
    return jp.array([i, i, i])


def apply_mass_props(model, ids: BodyIds, params: HopperParams, cargo: Cargo,
                     fuel_mass: jax.Array):
    m_pay = jp.maximum(cargo.mass, MIN_MASS)
    m_fuel = jp.maximum(fuel_mass, MIN_MASS)
    body_mass = model.body_mass.at[ids.payload].set(m_pay).at[ids.fuel].set(m_fuel)
    body_ipos = model.body_ipos.at[ids.payload].set(cargo.offset)
    body_inertia = (model.body_inertia
                    .at[ids.payload].set(point_inertia(m_pay, params.payload_radius))
                    .at[ids.fuel].set(point_inertia(m_fuel, params.fuel_radius)))
    return model.replace(body_mass=body_mass, body_ipos=body_ipos, body_inertia=body_inertia)


def loaded_com(params: HopperParams, cargo: Cargo, fuel_mass: jax.Array) -> jax.Array:
    """CoM of the loaded vehicle in the hopper body frame."""
    hull = params.dry_mass - params.gimbal_mass
    moment = (params.gimbal_mass * jp.array([0.0, 0.0, params.gimbal_z])
              + cargo.mass * cargo.offset
              + fuel_mass * jp.array([0.0, 0.0, params.tank_z]))
    return moment / (hull + params.gimbal_mass + cargo.mass + fuel_mass)
