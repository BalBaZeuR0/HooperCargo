import mujoco
import numpy as np

from hoopercargo.config import HopperParams
from hoopercargo.model import body_ids, build_model


def _body(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _geom_name(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)


def test_model_compiles_and_has_parts():
    m = build_model(1.0)
    for name in ["hopper", "payload", "fuel", "landing_pad", "launch_pad"]:
        assert _body(m, name) >= 0


def test_dry_mass_matches_haktan():
    m = build_model(1.0)
    p = HopperParams.for_scale(1.0)
    hop = _body(m, "hopper")
    skip = {_body(m, "payload"), _body(m, "fuel")}
    dry = sum(m.body_mass[b] for b in range(m.nbody)
              if b not in skip and _in_subtree(m, b, hop))
    assert abs(dry - 1.63) < 1e-6
    assert abs(p.dry_mass - 1.63) < 1e-9


def _in_subtree(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def test_only_feet_collide_with_ground():
    m = build_model(1.0)
    colliding = {_geom_name(m, g) for g in range(m.ngeom)
                 if m.geom_contype[g] or m.geom_conaffinity[g]}
    assert colliding == {"ground", "foot1_geom", "foot2_geom", "foot3_geom", "foot4_geom"}


def test_feet_rest_on_ground_at_start():
    m = build_model(1.0)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ids = body_ids(m)
    foot_bottom = d.geom_xpos[list(ids.foot_geoms), 2] - m.geom_size[list(ids.foot_geoms), 0]
    assert np.allclose(foot_bottom, 0.0, atol=1e-6)


def test_scale2_mass_x8_and_same_thrust_to_weight():
    p1, p2 = HopperParams.for_scale(1.0), HopperParams.for_scale(2.0)
    m1, m2 = build_model(1.0), build_model(2.0)
    assert np.isclose(m2.body_mass.sum(), 8 * m1.body_mass.sum())
    assert np.isclose(p2.max_thrust / (p2.dry_mass + p2.fuel_mass0),
                      p1.max_thrust / (p1.dry_mass + p1.fuel_mass0))
    assert np.isclose(p2.leg_span, 2 * p1.leg_span)
    hop1, hop2 = _body(m1, "hopper"), _body(m2, "hopper")
    assert np.allclose(m2.body_inertia[hop2], 32 * m1.body_inertia[hop1])
