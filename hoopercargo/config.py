"""Physical constants of the cargo hopper and the training curriculum.

Every value is given for scale = 1 (Haktan's ~1 m hopper) and rescaled with
`HopperParams.for_scale` using Froude similarity (L = scale): lengths x L,
masses x L^3, inertias x L^5, forces x L^3, areas x L^2, speeds and times
x sqrt(L).  The control step `dt` is kept fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

G0 = 9.80665
AIR_DENSITY = 1.225


@dataclass(frozen=True)
class HopperParams:
    scale: float = 1.0
    # mass budget (kg)
    dry_mass: float = 1.63            # hopper 1.6 + TVC gimbal 0.03
    gimbal_mass: float = 0.03
    gimbal_z: float = -0.15
    fuel_mass0: float = 0.6
    payload_mass_max: float = 0.8
    # propulsion
    max_thrust: float = 45.0          # N
    max_tvc_rad: float = math.radians(20.0)
    isp: float = 150.0                # s
    servo_speed: float = math.radians(60.0 / 0.13)   # rad/s
    # geometry (m, body frame, origin = dry CoM)
    bay_z_min: float = 0.0
    bay_z_max: float = 0.35
    bay_lateral_max: float = 0.03
    tank_z: float = -0.07
    payload_radius: float = 0.04
    fuel_radius: float = 0.05
    leg_span: float = 0.8
    foot_radius: float = 0.025
    rest_height: float = 0.38         # dry CoM height with feet on the ground
    pad_radius: float = 1.5
    airborne_height: float = 0.3      # lowest foot above this -> took off
    contact_tol: float = 0.01
    # aerodynamics
    drag_area: float = 0.072          # m^2
    drag_cd: float = 1.0
    ang_damping: float = 0.34         # N m s / rad (Haktan's value)
    # landing criteria
    touchdown_vz_max: float = 2.0     # m/s
    touchdown_vxy_max: float = 1.0
    settle_time: float = 2.0          # s
    tilt_success_rad: float = math.radians(15.0)
    tilt_fail_rad: float = math.radians(60.0)
    episode_seconds: float = 40.0
    # simulation
    timestep: float = 0.002
    frame_skip: int = 10

    @property
    def dt(self) -> float:
        return self.timestep * self.frame_skip

    @property
    def episode_steps(self) -> int:
        return int(round(self.episode_seconds / self.dt))

    @classmethod
    def for_scale(cls, scale: float = 1.0) -> "HopperParams":
        L = float(scale)
        b = cls()
        s = math.sqrt(L)
        return replace(
            b,
            scale=L,
            dry_mass=b.dry_mass * L**3,
            gimbal_mass=b.gimbal_mass * L**3,
            gimbal_z=b.gimbal_z * L,
            fuel_mass0=b.fuel_mass0 * L**3,
            payload_mass_max=b.payload_mass_max * L**3,
            max_thrust=b.max_thrust * L**3,
            isp=b.isp * s,
            servo_speed=b.servo_speed / s,
            bay_z_min=b.bay_z_min * L,
            bay_z_max=b.bay_z_max * L,
            bay_lateral_max=b.bay_lateral_max * L,
            tank_z=b.tank_z * L,
            payload_radius=b.payload_radius * L,
            fuel_radius=b.fuel_radius * L,
            leg_span=b.leg_span * L,
            foot_radius=b.foot_radius * L,
            rest_height=b.rest_height * L,
            pad_radius=b.pad_radius * L,
            airborne_height=b.airborne_height * L,
            contact_tol=b.contact_tol * L,
            drag_area=b.drag_area * L**2,
            ang_damping=b.ang_damping * L**4.5,
            touchdown_vz_max=b.touchdown_vz_max * s,
            touchdown_vxy_max=b.touchdown_vxy_max * s,
            settle_time=b.settle_time * s,
            episode_seconds=b.episode_seconds * s,
        )


@dataclass(frozen=True)
class StageConfig:
    dist_min: float
    dist_max: float
    payload_max: float
    wind_max: float
    gust_sigma: float

    def scaled(self, params: HopperParams) -> "StageConfig":
        L = params.scale
        s = math.sqrt(L)
        return StageConfig(
            dist_min=self.dist_min * L,
            dist_max=self.dist_max * L,
            payload_max=self.payload_max * L**3,
            wind_max=self.wind_max * s,
            gust_sigma=self.gust_sigma * s,
        )


# Curriculum (scale = 1 units): short/light/calm -> full task.
STAGES: tuple[StageConfig, ...] = (
    StageConfig(dist_min=5.0, dist_max=20.0, payload_max=0.3, wind_max=0.0, gust_sigma=0.0),
    StageConfig(dist_min=20.0, dist_max=60.0, payload_max=0.8, wind_max=2.0, gust_sigma=0.5),
    StageConfig(dist_min=20.0, dist_max=100.0, payload_max=0.8, wind_max=6.0, gust_sigma=1.5),
)
