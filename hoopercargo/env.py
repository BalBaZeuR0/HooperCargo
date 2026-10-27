"""Pure-JAX MJX environment: fly the cargo hopper to a random pad and land upright.

Physics structure (TVC servo, thrust and torques through `xfrc_applied`)
follows mjx/env.py of GulogluHaktan/3DRocketRl.
"""
from __future__ import annotations

from typing import Any

from brax.envs import base
from flax import struct
import jax
import jax.numpy as jp
from mujoco import mjx

from .config import G0, STAGES, HopperParams
from .model import body_ids, build_model
from .payload import Cargo, apply_mass_props, sample_cargo
from .reward import event_reward, guidance_velocity, step_reward
from .wind import WindState, drag_force, sample_wind, step_gust

OBS_SIZE = 31
METRIC_KEYS = ("success", "fail_tilt", "fail_offpad", "fail_hard", "fail_timeout",
               "landing_error", "fuel_used", "flight_time", "touch_vz", "touch_vxy")


@struct.dataclass
class CargoState:
    data: Any
    cargo: Cargo
    fuel: jax.Array
    wind: WindState
    gust_sigma: jax.Array
    pad_xy: jax.Array
    tvc_cmd: jax.Array
    last_action: jax.Array
    airborne: jax.Array
    touched: jax.Array
    touch_vz: jax.Array
    touch_vxy: jax.Array
    settle_timer: jax.Array
    step: jax.Array
    rng: jax.Array


class HooperCargoEnv(base.Env):
    """Cargo hopper point-to-point flight with random payload, fuel burn and wind."""

    def __init__(self, stage: int = 2, scale: float = 1.0, randomize_payload: bool = True,
                 observe_payload: bool = True):
        self.params = HopperParams.for_scale(scale)
        self.stage = STAGES[stage].scaled(self.params)
        self.randomize_payload = bool(randomize_payload)
        # a policy trained on a fixed payload sees a constant mass input, also when evaluated
        self.observe_payload = bool(observe_payload)
        self._host_model = build_model(scale)
        self._model = mjx.put_model(self._host_model)
        self._ids = body_ids(self._host_model)
        self._foot_geoms = jp.array(self._ids.foot_geoms)
        self.dt = self.params.dt

    @property
    def observation_size(self) -> int:
        return OBS_SIZE

    @property
    def action_size(self) -> int:
        return 3

    @property
    def backend(self) -> str:
        return "mjx"

    @property
    def host_model(self):
        return self._host_model

    @property
    def ids(self):
        return self._ids

    def model_for(self, cargo: Cargo, fuel: jax.Array):
        return apply_mass_props(self._model, self._ids, self.params, cargo, fuel)

    def _fixed_cargo(self) -> Cargo:
        p = self.params
        return Cargo(mass=jp.array(0.5 * p.payload_mass_max),
                     offset=jp.array([0.0, 0.0, 0.5 * (p.bay_z_min + p.bay_z_max)]))

    def reset(self, rng: jax.Array) -> base.State:
        p, st = self.params, self.stage
        rng, k_pad_d, k_pad_a, k_cargo, k_wind, k_yaw = jax.random.split(rng, 6)
        dist = jax.random.uniform(k_pad_d, (), minval=st.dist_min, maxval=st.dist_max)
        ang = jax.random.uniform(k_pad_a, (), minval=0.0, maxval=2.0 * jp.pi)
        pad_xy = dist * jp.array([jp.cos(ang), jp.sin(ang)])
        cargo = sample_cargo(k_cargo, p, st) if self.randomize_payload else self._fixed_cargo()
        fuel = jp.array(p.fuel_mass0)

        yaw = jax.random.uniform(k_yaw, (), minval=0.0, maxval=2.0 * jp.pi)
        qpos = jp.array(self._host_model.qpos0)
        qpos = qpos.at[3:7].set(jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        model = self.model_for(cargo, fuel)
        data = mjx.make_data(model).replace(qpos=qpos)
        data = data.replace(mocap_pos=data.mocap_pos
                            .at[self._ids.landing_pad_mocap].set(jp.array([pad_xy[0], pad_xy[1], 0.0]))
                            .at[self._ids.launch_pad_mocap].set(jp.zeros(3)))
        data = mjx.forward(model, data)

        state = CargoState(
            data=data, cargo=cargo, fuel=fuel,
            wind=sample_wind(k_wind, st), gust_sigma=jp.array(st.gust_sigma),
            pad_xy=pad_xy, tvc_cmd=jp.zeros(2), last_action=jp.zeros(3),
            airborne=jp.array(False), touched=jp.array(False),
            touch_vz=jp.array(0.0), touch_vxy=jp.array(0.0), settle_timer=jp.array(0.0),
            step=jp.array(0, dtype=jp.int32), rng=rng,
        )
        metrics = {k: jp.array(0.0) for k in METRIC_KEYS}
        return base.State(pipeline_state=state, obs=self._observe(state), reward=jp.array(0.0),
                          done=jp.array(0.0), metrics=metrics, info={})

    def _physics_substep(self, carry, _, *, model, throttle, tvc_target, thrust_on, wind_vel):
        data, tvc_cmd, fuel = carry
        p, ids = self.params, self._ids
        max_delta = p.servo_speed * p.timestep
        tvc_cmd = tvc_cmd + jp.clip(tvc_target - tvc_cmd, -max_delta, max_delta)
        data = data.replace(ctrl=tvc_cmd)

        # kinematics from the previous substep (one 2 ms lag; saves a forward pass)
        on = thrust_on & (fuel > 0.0)
        thrust = jp.where(on, throttle * p.max_thrust, 0.0)
        force = data.site_xmat[ids.thrust_site] @ jp.array([0.0, 0.0, 1.0]) * thrust
        body_com = data.xipos[ids.hopper]
        torque = jp.cross(data.site_xpos[ids.thrust_site] - body_com, force)

        drag, aero_torque = self.aero_wrench(data, wind_vel)
        torque += aero_torque
        omega_world = data.xmat[ids.hopper] @ data.qvel[3:6]
        torque -= p.ang_damping * omega_world

        xfrc = jp.zeros_like(data.xfrc_applied)
        xfrc = xfrc.at[ids.hopper, 0:3].set(force + drag).at[ids.hopper, 3:6].set(torque)
        data = mjx.step(model, data.replace(xfrc_applied=xfrc))
        fuel = jp.maximum(fuel - thrust / (p.isp * G0) * p.timestep, 0.0)
        return (data, tvc_cmd, fuel), None

    def step(self, state: base.State, action: jax.Array) -> base.State:
        s: CargoState = state.pipeline_state
        p = self.params
        action = jp.clip(action, -1.0, 1.0)
        throttle = (action[0] + 1.0) * 0.5
        tvc_target = action[1:3] * p.max_tvc_rad

        rng, k_gust = jax.random.split(s.rng)
        gust = step_gust(k_gust, s.wind.gust, s.gust_sigma, p.dt)
        wind = s.wind.replace(gust=gust)
        model = self.model_for(s.cargo, s.fuel)

        def substep(carry, x):
            return self._physics_substep(carry, x, model=model, throttle=throttle,
                                         tvc_target=tvc_target, thrust_on=~s.touched,
                                         wind_vel=wind.mean + wind.gust)

        (data, tvc_cmd, fuel), _ = jax.lax.scan(substep, (s.data, s.tvc_cmd, s.fuel), None,
                                                length=p.frame_skip)
        prev = s
        s = s.replace(data=data, tvc_cmd=tvc_cmd, fuel=fuel, wind=wind, rng=rng,
                      last_action=action, step=s.step + 1)
        s, reward, done, metrics = self._landing(prev, s, throttle)
        return state.replace(pipeline_state=s, obs=self._observe(s), reward=reward,
                             done=done.astype(jp.float32), metrics={**state.metrics, **metrics})

    def _landing(self, prev: CargoState, s: CargoState, throttle: jax.Array):
        """Take-off / touchdown bookkeeping, termination, reward and metrics."""
        p, d = self.params, s.data
        foot_h = self.foot_heights(d)
        contacts = foot_h <= p.contact_tol
        airborne = s.airborne | (jp.min(foot_h) > p.airborne_height)
        first_touch = airborne & ~s.touched & jp.any(contacts)
        # velocity entering the step: contact impulses have not damped it yet
        v_prev = prev.data.qvel[0:3]
        touch_vz = jp.where(first_touch, jp.abs(v_prev[2]), s.touch_vz)
        touch_vxy = jp.where(first_touch, jp.linalg.norm(v_prev[0:2]), s.touch_vxy)
        touched = s.touched | first_touch
        settle_timer = jp.where(touched, s.settle_timer + p.dt, 0.0)

        tilt = self.tilt(d)
        com = d.subtree_com[self._ids.hopper]
        feet_xy = d.geom_xpos[self._foot_geoms, 0:2]
        feet_in_pad = jp.all(jp.linalg.norm(feet_xy - s.pad_xy, axis=1) <= p.pad_radius)
        hard = (touch_vz >= p.touchdown_vz_max) | (touch_vxy >= p.touchdown_vxy_max)
        settled = touched & (settle_timer >= p.settle_time - 1e-6)

        tipped = tilt > p.tilt_fail_rad
        out = ((jp.linalg.norm(com[0:2] - s.pad_xy) > self.stage.dist_max + 30.0 * p.scale)
               | (com[2] > 60.0 * p.scale))
        upright_ok = jp.all(contacts) & (tilt < p.tilt_success_rad)
        success = settled & feet_in_pad & upright_ok & ~hard & ~tipped
        fail_offpad = (settled & ~feet_in_pad) | (out & ~tipped)
        fail_tilt = tipped | (settled & feet_in_pad & ~upright_ok)
        fail_hard = settled & feet_in_pad & upright_ok & hard & ~tipped
        terminal = success | fail_offpad | fail_tilt | fail_hard
        fail_timeout = ~terminal & (s.step >= p.episode_steps)
        done = terminal | fail_timeout

        rel_xy = s.pad_xy - com[0:2]
        v_des = guidance_velocity(rel_xy, jp.min(foot_h), p)
        shaping = step_reward(d.qvel[0:3], v_des, tilt, throttle, s.last_action - prev.last_action, p)
        reward = (jp.where(prev.touched, 0.0, shaping)
                  + event_reward(success, done & ~success))

        f = done.astype(jp.float32)
        metrics = {
            "success": f * success,
            "fail_tilt": f * fail_tilt,
            "fail_offpad": f * fail_offpad,
            "fail_hard": f * fail_hard,
            "fail_timeout": f * fail_timeout,
            "landing_error": f * jp.linalg.norm(jp.mean(feet_xy, axis=0) - s.pad_xy),
            "fuel_used": f * (1.0 - s.fuel / p.fuel_mass0),
            "flight_time": f * s.step * p.dt,
            "touch_vz": f * touch_vz,
            "touch_vxy": f * touch_vxy,
        }
        s = s.replace(airborne=airborne, touched=touched, touch_vz=touch_vz, touch_vxy=touch_vxy,
                      settle_timer=settle_timer)
        return s, reward, done, {k: jp.asarray(v, jp.float32) for k, v in metrics.items()}

    # ------------------------------------------------------------------ helpers
    def aero_wrench(self, data, wind_vel: jax.Array):
        """Drag force and its torque about the hopper body CoM (`xipos`, where xfrc acts).

        Drag acts at the centre of pressure (geometric centre of the hull), so wind
        tilts the vehicle when the loaded CoM sits below it.
        """
        hop = self._ids.hopper
        drag = drag_force(data.qvel[0:3], wind_vel, self.params)
        cp = data.xpos[hop] + data.xmat[hop] @ jp.array([0.0, 0.0, self.params.cp_z])
        return drag, jp.cross(cp - data.xipos[hop], drag)

    def foot_heights(self, data) -> jax.Array:
        return data.geom_xpos[self._foot_geoms, 2] - self.params.foot_radius

    def foot_contacts(self, data) -> jax.Array:
        return self.foot_heights(data) <= self.params.contact_tol

    def tilt(self, data) -> jax.Array:
        return jp.arccos(jp.clip(data.xmat[self._ids.hopper][2, 2], -1.0, 1.0))

    def _observe(self, s: CargoState) -> jax.Array:
        p, d = self.params, s.data
        L, sq = p.scale, jp.sqrt(p.scale)
        com = d.subtree_com[self._ids.hopper]
        rel_pad = jp.array([s.pad_xy[0], s.pad_xy[1], 0.0]) - com
        tvc = jp.array([d.qpos[self._ids.yaw_qpos], d.qpos[self._ids.pitch_qpos]])
        return jp.concatenate([
            rel_pad / (50.0 * L),
            d.qvel[0:3] / (10.0 * sq),
            d.xmat[self._ids.hopper].reshape(-1),
            d.qvel[3:6],
            tvc / p.max_tvc_rad,
            (s.fuel / p.fuel_mass0)[None],
            jp.where(self.observe_payload, s.cargo.mass / p.payload_mass_max, 0.5)[None],
            self.foot_contacts(d).astype(jp.float32),
            s.airborne.astype(jp.float32)[None],
            s.touched.astype(jp.float32)[None],
            s.last_action,
        ])
