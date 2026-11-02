"""Hand-written cascaded PD controller, used as a guide for jump-start training.

Outer loop: position/velocity error -> desired acceleration (hold a cruise
height while travelling, descend slowly once over the pad). Inner loop: the
thrust direction that gives that acceleration -> body attitude error -> gimbal
angles. Untuned; it lands about a quarter of stage-0 flights, which is enough
for PPO to see successful landings from the start of training.
"""
from __future__ import annotations

import jax
import jax.numpy as jp

from .config import G0
from .reward import TAKEOFF_HEIGHT


def pd_action(env, s, model) -> jax.Array:
    """Normalised action [throttle, gimbal_a, gimbal_b] in [-1, 1] for pipeline state `s`."""
    p, ids, d = env.params, env.ids, s.data
    L, sq = p.scale, jp.sqrt(p.scale)
    # loaded mass; MJX does not refresh body_subtreemass when payload/fuel masses change
    m = p.dry_mass + s.cargo.mass + s.fuel
    com = d.subtree_com[ids.hopper]
    v = d.qvel[0:3]
    rot = d.xmat[ids.hopper]
    e_xy = s.pad_xy - com[0:2]
    over = (jp.linalg.norm(e_xy) < 0.4 * L) & (jp.linalg.norm(v[0:2]) < 0.4 * sq)
    h = com[2] - p.rest_height
    # same take-off gate as the guidance field: climb first, then lean toward the pad
    gate = jp.clip(h / (TAKEOFF_HEIGHT * L), 0.0, 1.0)
    a_xy = jp.clip(0.6 * e_xy - 1.2 * v[0:2], -2.5, 2.5) * gate
    late = s.step > (0.6 * p.episode_steps)
    vz_des = jp.where(over | late,
                      -jp.clip(0.5 * h + 0.2 * sq, 0.2 * sq, 0.6 * sq),
                      jp.clip(1.0 * (3.0 * L - h), -1.0 * sq, 1.5 * sq))
    a_z = 2.0 * (vz_des - v[2])
    force = m * jp.array([a_xy[0], a_xy[1], a_z + G0])
    z_des = force / jp.linalg.norm(force)
    z_b = rot[:, 2]
    err_body = rot.T @ jp.cross(z_b, z_des)
    tau = (8.0 * err_body - 1.5 * d.qvel[3:6]) * L**4
    thrust = jp.clip(jp.dot(force, z_b), 0.0, p.max_thrust)
    arm = 0.25 * L * jp.maximum(thrust, 5.0 * L**3)
    lim = jp.sin(p.max_tvc_rad)
    a = jp.arcsin(jp.clip(-tau[0] / arm, -lim, lim))
    b = jp.arcsin(jp.clip(-tau[1] / arm, -lim, lim))
    return jp.array([2.0 * thrust / p.max_thrust - 1.0, a / p.max_tvc_rad, b / p.max_tvc_rad])
