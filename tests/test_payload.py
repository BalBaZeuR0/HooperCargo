import jax
import jax.numpy as jnp
from mujoco import mjx

from hoopercargo.config import STAGES, HopperParams
from hoopercargo.model import body_ids, build_model
from hoopercargo.payload import Cargo, apply_mass_props, loaded_com, sample_cargo

P = HopperParams.for_scale(1.0)
KEY = jax.random.PRNGKey(0)


def test_sampled_cargo_inside_bay():
    cs = jax.vmap(lambda k: sample_cargo(k, P, STAGES[2]))(jax.random.split(KEY, 4096))
    assert (cs.offset[:, 2] >= 0.0).all() and (cs.offset[:, 2] <= 0.35).all()
    assert (jnp.linalg.norm(cs.offset[:, :2], axis=1) <= 0.03 + 1e-9).all()
    assert (cs.mass >= 0).all() and (cs.mass <= 0.8).all()
    # covers the whole bay, not a corner of it
    assert float(cs.offset[:, 2].max()) > 0.3 and float(cs.mass.max()) > 0.7


def test_loaded_com_matches_hand_calc():
    c = Cargo(mass=jnp.array(0.8), offset=jnp.array([0.03, 0.0, 0.30]))
    com = loaded_com(P, c, jnp.array(0.6))
    M = 1.63 + 0.8 + 0.6
    # 0.03 kg TVC gimbal bodies sit at z = -0.15; the hopper's own mass is at the origin
    expected = jnp.array([0.8 * 0.03, 0.0, 0.8 * 0.30 + 0.6 * (-0.07) + 0.03 * (-0.15)]) / M
    assert jnp.allclose(com, expected, atol=1e-4)


def test_mjx_com_matches_loaded_com():
    m = build_model(1.0)
    ids = body_ids(m)
    mx = mjx.put_model(m)
    c = Cargo(mass=jnp.array(0.8), offset=jnp.array([0.03, -0.01, 0.30]))
    mx2 = apply_mass_props(mx, ids, P, c, jnp.array(0.6))
    d = mjx.forward(mx2, mjx.make_data(mx2))
    got = d.subtree_com[ids.hopper] - d.xpos[ids.hopper]
    assert jnp.allclose(got, loaded_com(P, c, jnp.array(0.6)), atol=2e-3)


def test_zero_fuel_keeps_positive_mass():
    m = build_model(1.0)
    mx = mjx.put_model(m)
    mx2 = apply_mass_props(mx, body_ids(m), P, Cargo(mass=jnp.array(0.0), offset=jnp.zeros(3)),
                           jnp.array(0.0))
    assert (mx2.body_mass[1:] > 0).all()
    assert (mx2.body_inertia[1:] > 0).all()
