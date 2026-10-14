"""
BR-1 Gymnasium environment (Milestone 3).

    Observation -> RL Policy -> Action -> PD Controller -> MuJoCo Physics
         ^                                                        |
         '------------------------- Sensors <----------------------'

`step(action)` runs `control_decimation` physics steps (500 Hz sim /
decimation 10 = 50 Hz control, config/robot.yaml) per RL step. The action is
a 12-dim offset from the nominal standing pose (Part 8: residual position
action) converted to torque by the SAME PDController class used in
Milestone 2 -- but here the policy owns all 12 joint targets, including the
ankles. There is no hard-coded balance controller in this env: learning to
not fall is exactly the Milestone 4 task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import yaml

from control.actuator import ActuatorInterface
from control.pd_controller import PDController
from control.balance_controller import POSTURE_GAINS
from robot.sensors import SensorNoise
from env.observations import build_observation, projected_gravity, OBS_DIM
from env.rewards import compute_reward
from env.termination import check_termination

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO / "mujoco" / "scene.xml"
DEFAULT_TRAINING_CFG = REPO / "config" / "training.yaml"


class BipedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene_path=DEFAULT_SCENE, training_cfg_path=DEFAULT_TRAINING_CFG,
                 use_sensor_noise: bool | None = None, seed: int | None = None):
        super().__init__()
        with open(training_cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.act = ActuatorInterface(self.model)
        kp, kd = self.act.gains_by_suffix(POSTURE_GAINS)
        self.pd = PDController(kp, kd, self.act.torque_limit)

        self.kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self.q_nominal = self.model.key_qpos[self.kid][self.act.qpos_adr].copy()
        self.height_target = float(self.model.key_qpos[self.kid][2])

        env_cfg = self.cfg["env"]
        self.decimation = int(env_cfg["control_decimation"])
        self.dt_ctrl = self.model.opt.timestep * self.decimation
        self.action_scale = float(env_cfg["action_scale_rad"])
        self.init_qpos_noise = float(env_cfg["init_qpos_noise_rad"])
        self.max_episode_steps = int(round(env_cfg["episode_length_s"] / self.dt_ctrl))
        self.max_tilt_deg = float(env_cfg["termination"]["max_tilt_deg"])
        self.min_base_height = float(env_cfg["termination"]["min_base_height_m"])
        self.cmd_ranges = env_cfg["command_ranges"]
        use_noise = env_cfg.get("use_sensor_noise", True) if use_sensor_noise is None else use_sensor_noise

        reward_cfg = self.cfg["reward"]
        self.reward_weights = reward_cfg["weights"]
        self.tracking_sigma = float(reward_cfg["tracking_sigma"])
        # Milestone 6: optional, narrower kernel for forward-velocity tracking
        # alone. With the shared 0.25 width, standing still at a 0.2 m/s
        # command keeps 85% of lin_vel_track, so EXP-005 never learned to
        # modulate speed. Defaults to tracking_sigma (M3-M5 behaviour).
        self.lin_vel_sigma = float(reward_cfg.get("lin_vel_sigma", self.tracking_sigma))
        self.fall_penalty = float(reward_cfg["fall_penalty"])

        # Milestone 5: gait clock. EXP-003's first try advanced the phase at
        # (cmd_speed) Hz -- at a 0.05 m/s command that is a 20 s step period,
        # i.e. no usable timing signal. Stepping cadence is now a fixed
        # frequency, switched on whenever the command asks for motion; at
        # cmd=0 (stage 'stand') the clock stays frozen exactly as before.
        gait_cfg = self.cfg.get("gait", {})
        self.gait_freq_hz = float(gait_cfg.get("frequency_hz", 1.5))
        self.gait_min_cmd = float(gait_cfg.get("min_cmd_speed", 0.05))
        self.swing_height = float(gait_cfg.get("swing_height_m", 0.05))

        # Milestone 7 (robustness), both off unless configured -- earlier
        # stages are unchanged. domain_rand re-draws physical parameters every
        # episode around the nominal model; pushes kick the base velocity at
        # random times. All ranges are INITIAL ASSUMPTIONs sized to cover the
        # uncertainty tags in config/robot.yaml, not measured hardware spread.
        self.dr_cfg = self.cfg.get("domain_rand") or {}
        self.push_cfg = self.cfg.get("pushes") or {}
        self._nominal_friction = self.model.geom_friction[:, 0].copy()
        self._torso_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._nominal_torso_mass = float(self.model.body_mass[self._torso_body])
        self._nominal_torso_inertia = self.model.body_inertia[self._torso_body].copy()
        self._motor_strength = np.ones(self.act.n)
        self._next_push_step = -1

        self.rng = np.random.default_rng(seed)
        self.noise = SensorNoise(self.rng) if use_noise else None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.act.n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self._prev_action = np.zeros(self.act.n)
        self._prev_qvel = np.zeros(self.act.n)
        self._cmd = np.zeros(3)
        self._clock_phase = 0.0
        self._step_count = 0

        self._foot_body = {
            "L": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "L_foot"),
            "R": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "R_foot"),
        }
        self._foot_geom = {
            "L": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "L_sole"),
            "R": mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "R_sole"),
        }
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # ------------------------------------------------------------------ #
    def _sample_command(self) -> np.ndarray:
        def u(pair):
            lo, hi = pair
            return self.rng.uniform(lo, hi) if hi > lo else lo
        return np.array([u(self.cmd_ranges["lin_vel_x"]),
                          u(self.cmd_ranges["lin_vel_y"]),
                          u(self.cmd_ranges["ang_vel_z"])])

    def _randomize_physics(self) -> None:
        """Re-draw the M7 domain-randomization parameters (no-op if unset)."""
        m = self.model
        m.geom_friction[:, 0] = self._nominal_friction
        m.body_mass[self._torso_body] = self._nominal_torso_mass
        m.body_inertia[self._torso_body] = self._nominal_torso_inertia
        self._motor_strength = np.ones(self.act.n)
        dr = self.dr_cfg
        if "friction_scale" in dr:  # floor + soles: slide friction of every geom
            m.geom_friction[:, 0] = self._nominal_friction * self.rng.uniform(*dr["friction_scale"])
        if "torso_added_mass_kg" in dr:  # payload / mass-estimate error, inertia scaled along
            new_mass = self._nominal_torso_mass + self.rng.uniform(*dr["torso_added_mass_kg"])
            m.body_mass[self._torso_body] = new_mass
            m.body_inertia[self._torso_body] = (self._nominal_torso_inertia
                                                 * new_mass / self._nominal_torso_mass)
        if "motor_strength" in dr:  # per-joint torque-constant error
            self._motor_strength = self.rng.uniform(*dr["motor_strength"], size=self.act.n)

    def _schedule_push(self) -> None:
        lo, hi = self.push_cfg["interval_s"]
        self._next_push_step = self._step_count + int(round(self.rng.uniform(lo, hi) / self.dt_ctrl))

    def _tilt_deg(self) -> float:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
        return float(np.degrees(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0))))

    def _foot_contacts(self):
        """Returns (contact_bool_per_foot, foot_xy_vel_per_foot, non_foot_ground_hit)."""
        contact = {"L": False, "R": False}
        non_foot_hit = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self._floor_geom not in (g1, g2):
                continue
            other = g2 if g1 == self._floor_geom else g1
            if other == self._foot_geom["L"]:
                contact["L"] = True
            elif other == self._foot_geom["R"]:
                contact["R"] = True
            else:
                non_foot_hit = True
        # cvel = [angular(3), linear(3)] about the body, world-aligned frame
        vel = {side: self.data.cvel[self._foot_body[side]][3:5].copy() for side in ("L", "R")}
        return contact, vel, non_foot_hit

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if self.noise is not None:
                self.noise.rng = self.rng  # keep the noise generator on the reseeded stream
        if self.noise is not None:
            self.noise.reset()

        self._randomize_physics()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.kid)
        self.data.qpos[self.act.qpos_adr] += self.rng.uniform(
            -self.init_qpos_noise, self.init_qpos_noise, self.act.n)
        self.data.qpos[2] -= 0.020  # match the M1/M2 ground-clearance correction
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(self.act.n)
        _, qd = self.act.joint_state(self.data)
        self._prev_qvel = qd.copy()
        # options={"cmd": [vx, vy, wz]} pins the command (evaluation sweeps).
        if options and "cmd" in options:
            self._cmd = np.asarray(options["cmd"], dtype=np.float64).copy()
        else:
            self._cmd = self._sample_command()
        self._clock_phase = 0.0
        self._step_count = 0
        self._next_push_step = -1
        if self.push_cfg:
            self._schedule_push()

        return self._get_obs(), {"cmd": self._cmd.copy()}

    def _sole_clearance(self, side: str) -> float:
        """Height of the sole box's lowest corner above the floor (z=0).
        Not the ankle-frame height: heel/toe rocking lifts the ankle ~1 cm
        while the sole is still planted, which would count as a 'step'."""
        g = self._foot_geom[side]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        corners = corners * self.model.geom_size[g]
        z = self.data.geom_xpos[g][2] + corners @ self.data.geom_xmat[g].reshape(3, 3)[2]
        return float(z.min())

    def _stepping(self) -> bool:
        return float(np.linalg.norm(self._cmd[:2])) >= self.gait_min_cmd

    def _desired_stance(self) -> list[bool]:
        """[L, R] feet the clock wants on the ground: both while standing,
        left on the first half-cycle and right on the second while stepping."""
        if not self._stepping():
            return [True, True]
        left = np.sin(self._clock_phase) >= 0.0
        return [bool(left), not left]

    def _get_obs(self) -> np.ndarray:
        return build_observation(self.data, self.act, self.q_nominal, self._cmd,
                                  self._prev_action, self._clock_phase, self.noise)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        q_target = self.q_nominal + self.action_scale * action

        pushed = self._step_count == self._next_push_step
        if pushed:  # M7: instantaneous horizontal base-velocity kick
            v = self.push_cfg["max_vel_xy"]
            self.data.qvel[0:2] += self.rng.uniform(-v, v, size=2)
            self._schedule_push()

        torque_sum = np.zeros(self.act.n)
        for _ in range(self.decimation):
            q, qd = self.act.joint_state(self.data)
            tau = self.pd.compute(q, qd, q_target) * self._motor_strength
            self.act.apply(self.data, tau)
            mujoco.mj_step(self.model, self.data)
            torque_sum += tau
        torque_avg = torque_sum / self.decimation

        mujoco.mj_forward(self.model, self.data)  # refresh contacts/CoM for the new qpos
        q, qd = self.act.joint_state(self.data)
        tilt = self._tilt_deg()
        base_height = float(self.data.qpos[2])
        contact, foot_vel, non_foot_hit = self._foot_contacts()

        terminated, reason = check_termination(
            base_height, tilt, self.data.qpos, self.data.qvel,
            self.max_tilt_deg, self.min_base_height)
        fall = terminated and reason not in ("instability (velocity blow-up)", "nan")

        state = {
            "base_linvel": self.data.qvel[0:3].copy(),
            "base_angvel_z": float(self.data.qvel[5]),
            "cmd": self._cmd,
            "projected_gravity": projected_gravity(self.data),
            "base_height": base_height,
            "height_target": self.height_target,
            "torque": torque_avg,
            "torque_limit": self.act.torque_limit,
            "action": action,
            "prev_action": self._prev_action,
            "qvel": qd,
            "prev_qvel": self._prev_qvel,
            "dt": self.dt_ctrl,
            "foot_contact": [contact["L"], contact["R"]],
            "foot_xy_vel": [foot_vel["L"], foot_vel["R"]],
            "non_foot_contact": non_foot_hit,
            "terminated_by_fall": fall,
            "desired_stance": self._desired_stance(),
            "foot_height": [self._sole_clearance(s) for s in ("L", "R")],
            "swing_height": self.swing_height,
        }
        reward, reward_terms = compute_reward(
            state, self.reward_weights, self.tracking_sigma, self.fall_penalty,
            lin_vel_sigma=self.lin_vel_sigma)

        if self._stepping():
            self._clock_phase += 2 * np.pi * self.gait_freq_hz * self.dt_ctrl
        self._prev_action = action.copy()
        self._prev_qvel = qd.copy()
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = {
            "reward_terms": reward_terms,
            "termination_reason": reason,
            "tilt_deg": tilt,
            "base_height": base_height,
            "base_linvel": state["base_linvel"],  # privileged; logging/eval only
            "base_pos": self.data.qpos[0:3].copy(),
            "foot_contact": (contact["L"], contact["R"]),
            "foot_height": tuple(state["foot_height"]),
            "cmd": self._cmd.copy(),
            "pushed": pushed,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        pass
