"""Builds the MuJoCo model of the cargo hopper at any geometric scale."""
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import mujoco

TEMPLATE = Path(__file__).resolve().parents[1] / "assets" / "hooper_cargo.xml"

# attribute -> exponent of L, per element tag (Froude similarity)
_SCALING = {
    "geom": {"pos": 1, "fromto": 1, "size": 1},
    "body": {"pos": 1},
    "site": {"pos": 1, "size": 1},
    "inertial": {"pos": 1, "mass": 3, "diaginertia": 5},
    "joint": {"damping": 4.5, "armature": 5, "frictionloss": 4},
    "position": {"kp": 4, "forcerange": 4},
}


class BodyIds(NamedTuple):
    hopper: int
    payload: int
    fuel: int
    landing_pad_mocap: int
    launch_pad_mocap: int
    foot_geoms: tuple[int, int, int, int]
    thrust_site: int
    yaw_qpos: int
    pitch_qpos: int


def _scale_values(text: str, factor: float) -> str:
    return " ".join(repr(float(v) * factor) for v in text.split())


def scaled_xml(scale: float = 1.0) -> str:
    root = ET.parse(TEMPLATE).getroot()
    L = float(scale)
    for tag, attrs in _SCALING.items():
        for el in root.iter(tag):
            for attr, exp in attrs.items():
                if attr in el.attrib:
                    el.set(attr, _scale_values(el.get(attr), L**exp))
    # contact time constants follow sqrt(L)
    for el in root.iter("geom"):
        if "solref" in el.attrib:
            tc, damp = el.get("solref").split()
            el.set("solref", f"{float(tc) * math.sqrt(L)!r} {damp}")
    return ET.tostring(root, encoding="unicode")


def build_model(scale: float = 1.0) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(scaled_xml(scale))


def body_ids(m: mujoco.MjModel) -> BodyIds:
    def bid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)

    def qpos(joint):
        return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)])

    return BodyIds(
        hopper=bid("hopper"),
        payload=bid("payload"),
        fuel=bid("fuel"),
        landing_pad_mocap=int(m.body_mocapid[bid("landing_pad")]),
        launch_pad_mocap=int(m.body_mocapid[bid("launch_pad")]),
        foot_geoms=tuple(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"foot{i}_geom") for i in range(1, 5)
        ),
        thrust_site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "thrust_site"),
        yaw_qpos=qpos("tvc_yaw_joint"),
        pitch_qpos=qpos("tvc_pitch_joint"),
    )
