"""
env_3dof.py — SO-101 stage-1 environment, 3-DOF variant.

    TRAINED : shoulder_pan, shoulder_lift, wrist_flex
    FIXED   : elbow_flex, wrist_roll, gripper   (from configs/fixed_angles.yaml)

This is the 6-DOF env with the action and observation reduced to the three
trained joints, the OpenCV detector as the only backend, and nothing else
changed. Same dwell episodes, same action filter, same action-delay
randomisation, same physics randomisation, same collision reporting, same
success-state buffer.

OBSERVATION (14-dim, unchanged layout, three joints instead of six)
--------------------------------------------------------------------
    With N trained joints (N = len(env.trained_joints)):
    [0:N]      trained joint positions (raw radians)
    [N:2N]     trained joint velocities
    [2N:3N]    previous action
    [3N:3N+4]  detection  [u, v, size, visible]
    [3N+4]     normalised episode progress
    [14:17] last-seen memory [u, v, staleness]  — ONLY if
            env.last_seen_memory is true, which makes the observation 17-dim.
            Appended rather than inserted so 0:14 is unchanged either way.

    Raw units throughout, because VecNormalize(normalize_obs=true) handles
    scaling. The 6-DOF layout was 23-dim with detection at [18:22]; the only
    change is 6 -> 3 in the first three blocks.

ACTION (N-dim, continuous, [-1, 1])
-----------------------------------
Delta position targets, scaled by max_delta and accumulated onto the actuator
setpoints. Deltas rather than absolute targets keep the commands smooth, which
matters for the STS3215 servos.

HOW THE FIXED JOINTS ARE HELD
-----------------------------
Position actuators stay in the model; their ctrl is rewritten to the fixed
value every step. They are not welded with <equality>. On the real arm these
are STS3215 servos under position control with finite holding stiffness that
sag under load; a weld would be perfectly rigid and only in simulation.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from rate import resolve as resolve_rate
from reward_3dof import Ctx, RewardWeights, Stage1Reward, SuccessCriteria
from vision.cv_detector import CubeDetector, DetectorConfig, MujocoCameraSource

HERE = Path(__file__).resolve().parent

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper"]
# Defaults; overridden by env.trained_joints. Anything not trained is fixed.
TRAINED = ["shoulder_pan", "shoulder_lift", "wrist_flex"]
FIXED = ["elbow_flex", "wrist_roll", "gripper"]

PAD_OFFSET = np.array([-0.0069, 0.0, -0.0880])   # pinch point, gripper frame
TIP_OFFSET = np.array([-0.0160, 0.0, -0.1044])   # fingertip — the lowest point
APPROACH_LOCAL = np.array([0.0, 0.0, -1.0])      # gripper -Z is the approach

# Geoms the arm must not touch. Cube-floor contact is excluded elsewhere.
OBSTACLE_GEOMS = ["floor", "floor_geom",
                  "base_collision_body", "base_collision_plate"]


class SO101Approach3DOF(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, seed: int | None = None):
        super().__init__()
        # One knob, env.control_hz; every rate-dependent value is derived.
        # Idempotent, so train/evaluate/tools may all have called it already.
        self.cfg = cfg = resolve_rate(cfg, verbose=bool(cfg.get("verbose_rate")))
        ecfg = cfg["env"]
        # seed=None -> OS entropy, so unseeded envs genuinely differ. Each
        # SubprocVecEnv worker is given seed+rank, so training is already
        # varied; this matters for the tools, where a fixed base seed replays
        # the identical cube sequence every run.
        self.rng = np.random.default_rng(seed)

        scene = Path(ecfg["scene"])
        if not scene.is_absolute():
            scene = HERE / scene
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(ecfg["frame_skip"])
        self.control_hz = 1.0 / (self.model.opt.timestep * self.frame_skip)
        # physics_hz in the config drives every derived value, so a scene whose
        # timestep disagrees would silently invalidate all of them.
        want = float(ecfg.get("physics_hz", 600.0))
        got = 1.0 / self.model.opt.timestep
        # RELATIVE tolerance. Scene XMLs write the timestep as a rounded
        # decimal — 1/600 becomes "0.00166667", which reads back as 599.9988 Hz
        # — so an absolute 1e-6 comparison can never pass. 0.1% catches a real
        # mismatch (600 vs 500) while ignoring decimal truncation.
        if not math.isclose(want, got, rel_tol=1e-3):
            raise ValueError(
                f"env.physics_hz={want:g} but {scene.name} has timestep "
                f"{self.model.opt.timestep:.9g} s = {got:.6f} Hz "
                f"({100*abs(want-got)/got:.2f}% off). Fix whichever is wrong — "
                f"every rate-derived value depends on this.")
        self.max_steps = int(round(ecfg["episode_seconds"] * self.control_hz))
        self.episode_mode = ecfg.get("episode_mode", "dwell")
        if self.episode_mode not in ("dwell", "terminate_on_success"):
            raise ValueError(f"unknown episode_mode {self.episode_mode!r}")
        # Which joints the policy controls. Everything else is held at the
        # value in fixed_angles.yaml. Freeing elbow_flex makes the approach
        # pitch controllable again: three parallel pitch axes give a redundant
        # planar arm, so position and orientation can be satisfied together
        # instead of pitch falling out of the geometry.
        self.trained = list(ecfg.get("trained_joints", TRAINED))
        bad = [n for n in self.trained if n not in ARM_JOINTS]
        if bad:
            raise ValueError(f"unknown joints in env.trained_joints: {bad}")
        self.fixed = [n for n in ARM_JOINTS if n not in self.trained]
        n_act = len(self.trained)
        for key in ("home_qpos", "max_delta"):
            if len(ecfg[key]) != n_act:
                raise ValueError(
                    f"env.{key} has {len(ecfg[key])} entries but "
                    f"{n_act} joints are trained {self.trained}. They must "
                    f"match, in the same order.")

        self.action_filter = float(ecfg.get("action_filter", 0.0))
        self.blind = bool(ecfg.get("blind", False))
        # Appended to the observation, so indices 0:14 are unchanged and the
        # 14-dim layout still works with last_seen_memory: false.
        self.last_seen_memory = bool(ecfg.get("last_seen_memory", False))
        self.ground_clearance = float(ecfg.get("ground_clearance", 0.010))

        # ---------------- ids ----------------
        self.jid, self.aid, self.qadr = {}, {}, {}
        for n in ARM_JOINTS:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if j < 0 or a < 0:
                raise RuntimeError(f"joint/actuator '{n}' missing from {scene}")
            self.jid[n], self.aid[n] = j, a
            self.qadr[n] = self.model.jnt_qposadr[j]

        self.gripper_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ecfg.get("gripper_body", "gripper"))
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        if self.gripper_body < 0 or self.cube_body < 0:
            raise RuntimeError("body 'gripper' or 'cube' missing from the scene")
        self.cube_jnt = self.model.body_jntadr[self.cube_body]
        self.cube_qadr = self.model.jnt_qposadr[self.cube_jnt]

        gid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.cube_geom = gid("cube_geom")

        # Geoms belonging to bodies the arm can actually MOVE, i.e. everything
        # in the kinematic subtree below the first trained joint.
        #
        # Without this filter the collision check fires on the robot's own
        # base resting on the floor: a legitimate 25-micrometre resting
        # contact, present from step 0 of every episode, which charged a flat
        # penalty on 100% of steps forever. Constant cost, so it does not
        # change the optimum, but it buries the real signal — you cannot tell
        # a genuine floor strike from the robot simply standing there.
        first = self.model.jnt_bodyid[self.jid[self.trained[0]]]
        movable = set()
        for b in range(self.model.nbody):
            anc = b
            while anc > 0:
                if anc == first:
                    movable.add(b)
                    break
                anc = self.model.body_parentid[anc]
        self.movable_geoms = {g for g in range(self.model.ngeom)
                              if self.model.geom_bodyid[g] in movable}
        _found = [n for n in OBSTACLE_GEOMS if gid(n) >= 0]
        if cfg.get("verbose_rate"):
            print(f"  obstacles       {_found or 'NONE FOUND'};  "
                  f"{len(self.movable_geoms)} movable geoms can trigger the "
                  f"collision penalty")
        if not _found and cfg.get("verbose_rate"):
            print("  NOTE: none of the obstacle geoms "
                  f"{OBSTACLE_GEOMS} exist in this scene, so the contact-based "
                  "collision penalty can never fire. The height-based ground "
                  "penalty (reward.weights.w_ground) still works.")
        self.obstacle_geoms = {g for g in (gid(n) for n in OBSTACLE_GEOMS) if g >= 0}
        self.cyl_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          "drop_cylinder")

        # Spawn radii are measured from the shoulder_pan AXIS, not the world
        # origin. If the arm is not modelled at (0,0) — and many scenes offset
        # it onto a table — a radius measured from the origin puts the
        # keep-out circle in the wrong place, and raising radius_min moves the
        # cube off-centre instead of further away.
        mujoco.mj_forward(self.model, self.data)
        pan_jid = self.jid["shoulder_pan"]
        self.base_xy = self.data.xanchor[pan_jid][:2].copy()

        self.t_qadr = np.array([self.qadr[n] for n in self.trained])
        self.t_dadr = np.array([self.model.jnt_dofadr[self.jid[n]]
                                for n in self.trained])
        self.t_aid = [self.aid[n] for n in self.trained]
        self.jnt_range = np.array([self.model.jnt_range[self.jid[n]]
                                   for n in self.trained])
        self.t_lo, self.t_hi = self.jnt_range[:, 0], self.jnt_range[:, 1]

        # ---------------- fixed angles ----------------
        fa_path = Path(ecfg["fixed_angles"])
        if not fa_path.is_absolute():
            fa_path = HERE / fa_path
        if not fa_path.exists():
            raise FileNotFoundError(
                f"{fa_path} not found. Run tools/setup_fixed_angles.py first — "
                "the frozen elbow angle determines which radii are reachable "
                "at all, so there is no sensible default.")
        fa = yaml.safe_load(fa_path.read_text())
        missing = [n for n in self.fixed if n not in fa["fixed_joints"]]
        if missing:
            raise ValueError(
                f"{fa_path} has no value for {missing}, which env."
                f"trained_joints leaves fixed. Re-run set_angles.py, or add "
                f"them by hand.")
        # Extra keys are fine and ignored: a fixed_angles.yaml written when
        # elbow_flex was frozen stays valid after you free it.
        self.fixed_vals = {n: float(fa["fixed_joints"][n]) for n in self.fixed}
        for n, v in self.fixed_vals.items():
            lo, hi = self.model.jnt_range[self.jid[n]]
            if not (lo - 1e-6 <= v <= hi + 1e-6):
                raise ValueError(f"fixed {n}={v:.3f} outside limits "
                                 f"[{lo:.3f}, {hi:.3f}]")

        # ---------------- detector ----------------
        dcfg = dict(cfg["detector"])
        self.camera_name = ecfg.get("camera", "wrist_cam")
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                        self.camera_name)
        self.camera = MujocoCameraSource(self.model, camera=self.camera_name,
                                         width=int(dcfg["width"]),
                                         height=int(dcfg["height"]))
        self.detector = CubeDetector(DetectorConfig(**dcfg))

        # ---------------- reward ----------------
        rcfg = cfg["reward"]
        self.reward_fn = Stage1Reward(RewardWeights(**rcfg["weights"]),
                                      SuccessCriteria(**rcfg["success"]))

        # ---------------- spaces ----------------
        self.max_delta = np.array(ecfg["max_delta"], dtype=np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        # 3 blocks of n_act (qpos, qvel, prev_action) + detection 4 + time 1
        obs_dim = 3 * n_act + 5 + (3 if self.last_seen_memory else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,),
                                            np.float32)

        self.spawn = cfg["spawn"]
        self.rand = cfg.get("randomization", {})
        self.save_success_states = bool(
            cfg.get("train", {}).get("save_success_states", False))
        self.success_states: list = []

        # Nominals captured before any randomisation, so every reset perturbs the
        # original values rather than compounding.
        self._nom = dict(
            cam_pos=self.model.cam_pos[self.cam_id].copy() if self.cam_id >= 0 else None,
            cam_quat=self.model.cam_quat[self.cam_id].copy() if self.cam_id >= 0 else None,
            cam_fovy=float(self.model.cam_fovy[self.cam_id]) if self.cam_id >= 0 else None,
            gainprm=self.model.actuator_gainprm.copy(),
            biasprm=self.model.actuator_biasprm.copy(),
            dof_damping=self.model.dof_damping.copy(),
            geom_friction=self.model.geom_friction.copy(),
            body_mass=self.model.body_mass.copy(),
        )
        self.last_detection = np.zeros(4, dtype=np.float32)
        self.last_pinch_vel = np.zeros(3)
        self._seen_mem = np.zeros(3, dtype=np.float32)
        self._steps_unseen = 0

        if cfg.get("verbose_rate"):
            sp = cfg["spawn"]
            print(f"  spawn centre    ({self.base_xy[0]:+.3f}, "
                  f"{self.base_xy[1]:+.3f}) m — the shoulder_pan axis"
                  f"{'' if np.allclose(self.base_xy, 0, atol=1e-6) else '  (NOT the world origin)'}")
            print(f"  keep-out radius {sp['radius_min']:.3f} m; cubes spawn "
                  f"{sp['radius_min']:.3f}-{sp['radius_max']:.3f} m from it")

    # ==================================================================
    @property
    def pinch_pos(self) -> np.ndarray:
        R = self.data.xmat[self.gripper_body].reshape(3, 3)
        return self.data.xpos[self.gripper_body] + R @ PAD_OFFSET

    @property
    def tip_pos(self) -> np.ndarray:
        R = self.data.xmat[self.gripper_body].reshape(3, 3)
        return self.data.xpos[self.gripper_body] + R @ TIP_OFFSET

    def _ground_depth(self) -> float:
        """How far the lowest part of the gripper is below clearance, in m.

        Height-based, so it works regardless of what the floor geom is called.
        The name-based collision check above is the better signal when it
        fires, but it silently reports nothing if the scene uses different
        names — and a penalty that quietly does nothing is worse than none.
        """
        low = min(float(self.pinch_pos[2]), float(self.tip_pos[2]))
        return max(0.0, self.ground_clearance - low)

    @property
    def approach_axis(self) -> np.ndarray:
        return self.data.xmat[self.gripper_body].reshape(3, 3) @ APPROACH_LOCAL

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def _q_norm(self) -> np.ndarray:
        q = self.data.qpos[self.t_qadr]
        return 2.0 * (q - self.t_lo) / (self.t_hi - self.t_lo) - 1.0

    # ==================================================================
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.reward_fn.reset()

        for n, v in self.fixed_vals.items():
            self.data.qpos[self.qadr[n]] = v
            self.data.ctrl[self.aid[n]] = v

        home = np.array(self.cfg["env"]["home_qpos"], dtype=np.float64)
        home = np.clip(home + self.rng.normal(0, self.cfg["env"]["home_noise"],
                                              len(self.trained)),
                       self.t_lo, self.t_hi)
        self.data.qpos[self.t_qadr] = home
        self.data.ctrl[self.t_aid] = home
        self.target_ctrl = home.copy()

        self._spawn_cube()
        self._randomize()

        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)

        self.step_count = 0
        self.first_success_step = None
        self.prev_action = np.zeros(len(self.trained), dtype=np.float32)
        self._filt_action = np.zeros(len(self.trained), dtype=np.float32)
        self._prev_pinch = self.pinch_pos.copy()
        self._prev_qvel = self.data.qvel[self.t_dadr].copy()
        self.last_pinch_vel = np.zeros(3)
        self._seen_mem[:] = 0.0
        self._steps_unseen = 0

        det = self._detect()
        self._update_memory(det)
        return self._obs(det), {}

    def _spawn_cube(self):
        s = self.spawn
        for _ in range(200):
            r = self.rng.uniform(s["radius_min"], s["radius_max"])
            th = math.radians(self.rng.uniform(s["angle_min_deg"],
                                               s["angle_max_deg"]))
            x = self.base_xy[0] + r * math.cos(th)
            y = self.base_xy[1] + r * math.sin(th)
            # Keep the cube away from the drop cylinder, or a fraction of
            # episodes are unsolvable because the target is inside an obstacle.
            if self.cyl_body >= 0:
                cyl = self.model.body_pos[self.cyl_body]
                if math.hypot(x - cyl[0], y - cyl[1]) < s.get("min_clearance", 0.0):
                    continue
            break
        self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = [x, y, s["cube_half"]]
        yaw = self.rng.uniform(-math.pi, math.pi)
        self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = [
            math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

        # hidden_fraction: force a share of episodes to start with the cube out
        # of view by rotating the ARM away, not by moving the cube. Sampling
        # pan from the cube's azimuth in either direction would correlate
        # proprioception with the answer, which is the leak found in the 6-DOF
        # project — so the pan offset is drawn independently of th.
        if self.rng.random() < self.spawn.get("hidden_fraction", 0.0):
            self.data.qpos[self.t_qadr[0]] = self.rng.uniform(self.t_lo[0],
                                                              self.t_hi[0])
            self.target_ctrl[0] = self.data.qpos[self.t_qadr[0]]
            self.data.ctrl[self.t_aid[0]] = self.target_ctrl[0]

    def _randomize(self):
        rc = self.rand
        # Action delay is sampled every episode regardless, since 0 delay is a
        # valid draw and the queue must exist either way.
        lo = int(rc.get("action_delay_min", 0))
        hi = int(rc.get("action_delay_max", 0))
        n_delay = (0 if not rc.get("enabled", True)
                   else int(self.rng.integers(lo, hi + 1)))
        self._action_queue = [np.zeros(len(self.trained), dtype=np.float32)
                              for _ in range(n_delay)]
        if not rc.get("enabled", True):
            return

        m, rng, nom = self.model, self.rng, self._nom

        # Servo gains. The baked-in kp=998 assumes an idealised STS3215.
        f = 1.0 + rng.uniform(-rc["kp_frac"], rc["kp_frac"])
        m.actuator_gainprm[:] = nom["gainprm"] * f
        m.actuator_biasprm[:] = nom["biasprm"] * f
        m.dof_damping[:] = nom["dof_damping"] * (
            1.0 + rng.uniform(-rc["damping_frac"], rc["damping_frac"]))
        m.geom_friction[:] = nom["geom_friction"] * (
            1.0 + rng.uniform(-rc["friction_frac"], rc["friction_frac"]))

        m.body_mass[:] = nom["body_mass"]
        m.body_mass[self.cube_body] = nom["body_mass"][self.cube_body] * (
            1.0 + rng.uniform(-rc["cube_mass_frac"], rc["cube_mass_frac"]))
        if self.cube_geom >= 0:
            m.geom_friction[self.cube_geom, 0] = rng.uniform(
                *rc["cube_friction_range"])

        # Camera extrinsics: the printed mount will not match CAD exactly.
        if self.cam_id >= 0:
            m.cam_pos[self.cam_id] = nom["cam_pos"] + rng.normal(
                0, rc["cam_pos_mm"] / 1000.0, 3)
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            dq, out = np.empty(4), np.empty(4)
            mujoco.mju_axisAngle2Quat(dq, axis,
                                      math.radians(rng.normal(0, rc["cam_rot_deg"])))
            mujoco.mju_mulQuat(out, dq, nom["cam_quat"])
            m.cam_quat[self.cam_id] = out
            m.cam_fovy[self.cam_id] = nom["cam_fovy"] + rng.normal(
                0, rc["cam_fovy_deg"])

        self.detector.randomize(rng)

    # ==================================================================
    def _detect(self) -> np.ndarray:
        det = (np.zeros(4, dtype=np.float32) if self.blind
               else self.detector.detect(self.camera.frame(self.data)))
        self.last_detection = det
        return det

    def _collision(self) -> tuple[bool, float]:
        """Arm touching the floor or the base primitives, and how deep.

        Depth as well as a flag so the reward can charge grazing less than
        driving through. Cube contact is excluded: the other geom must belong
        to the robot.
        """
        hit, depth = False, 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            in1, in2 = g1 in self.obstacle_geoms, g2 in self.obstacle_geoms
            if in1 == in2:
                continue
            other = g2 if in1 else g1
            if other == self.cube_geom:
                continue
            # Only geoms the policy can move. The base sitting on the floor is
            # not a collision, and counting it makes the penalty a constant.
            if other not in self.movable_geoms:
                continue
            hit = True
            depth = max(depth, max(0.0, -float(c.dist)))
        return hit, depth

    def _update_memory(self, det: np.ndarray):
        """Remember where the cube was last seen, and how long ago.

        THIS IS WHAT MAKES SEARCH LEARNABLE. Without it, the moment the cube
        leaves frame the observation is byte-identical regardless of which
        side it exited, so no feed-forward policy can turn the right way — the
        best available strategy is a fixed sweep that is wrong half the time.
        Carrying the last in-frame (u, v) plus staleness turns that into a
        solvable problem without paying for a recurrent policy.

        Staleness is capped at 2.0 (60 steps) so a long search does not push
        the observation far outside the range VecNormalize has seen.
        """
        if det[3] > 0.5:
            self._seen_mem[:2] = det[:2]
            self._steps_unseen = 0
        else:
            self._steps_unseen += 1
        self._seen_mem[2] = min(self._steps_unseen / 30.0, 2.0)

    def _obs(self, det: np.ndarray) -> np.ndarray:
        parts = [
            self.data.qpos[self.t_qadr],
            self.data.qvel[self.t_dadr],
            self.prev_action,
            det,
            [self.step_count / self.max_steps],
        ]
        if self.last_seen_memory:
            parts.append(self._seen_mem)
        return np.concatenate(parts).astype(np.float32)

    def _out_of_bounds(self) -> bool:
        p = self.pinch_pos
        reach = float(self.cfg["spawn"].get("max_reach", 0.42))
        return bool(p[2] < 0.005
                    or np.linalg.norm(p[:2] - self.base_xy) > reach
                    or self.cube_pos[2] < 0.0)

    # ==================================================================
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        # Randomised actuation delay: a real USB-servo chain does not apply a
        # command the instant it is issued, and a policy trained without it
        # learns a control law that depends on zero latency.
        if self._action_queue:
            self._action_queue.append(action.astype(np.float32))
            applied = self._action_queue.pop(0).astype(np.float64)
        else:
            applied = action

        # First-order low-pass. Changes the dynamics, so it must match between
        # training and deployment.
        a = self.action_filter
        self._filt_action = a * self._filt_action + (1.0 - a) * applied
        applied = self._filt_action

        self.target_ctrl = np.clip(self.target_ctrl + applied * self.max_delta,
                                   self.t_lo, self.t_hi)
        self.data.ctrl[self.t_aid] = self.target_ctrl
        for n, v in self.fixed_vals.items():
            self.data.ctrl[self.aid[n]] = v

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        pinch = self.pinch_pos
        pinch_vel = (pinch - self._prev_pinch) * self.control_hz
        self._prev_pinch = pinch.copy()
        self.last_pinch_vel = pinch_vel
        qvel = self.data.qvel[self.t_dadr].copy()

        det = self._detect()
        hit, depth = self._collision()
        ctx = Ctx(cube_pos=self.cube_pos, pinch_pos=pinch, pinch_vel=pinch_vel,
                  approach_axis=self.approach_axis, detection=det,
                  action=action.astype(np.float32), prev_action=self.prev_action,
                  qvel=qvel, prev_qvel=self._prev_qvel, q_norm=self._q_norm(),
                  out_of_bounds=self._out_of_bounds(),
                  collision=hit, collision_depth=depth,
                  ground_depth=self._ground_depth())
        reward, in_pose, comps = self.reward_fn(ctx)

        self._prev_qvel = qvel
        self.prev_action = action.astype(np.float32)
        self.step_count += 1
        if in_pose and self.first_success_step is None:
            self.first_success_step = self.step_count

        if self.episode_mode == "dwell":
            # Never end early: the reward for HOLDING is the task.
            terminated = False
        else:
            terminated = bool(in_pose or ctx.out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)

        if in_pose and self.save_success_states and len(self.success_states) < 20_000:
            self.success_states.append(
                (self.data.qpos.copy(), self.data.qvel.copy()))

        reached = self.first_success_step is not None
        info = {"is_success": reached, "in_pose": in_pose,
                "detected": bool(det[3] > 0.5), "collision": hit,
                "ground": bool(self._ground_depth() > 0.0),
                "out_of_bounds": ctx.out_of_bounds,
                **{f"rc/{k}": v for k, v in comps.items()}}
        if terminated or truncated:
            info["episode_success"] = float(reached)
            info["dwell_steps"] = float(self.reward_fn.dwell_steps)
            info["dwell_frac"] = float(self.reward_fn.dwell_steps
                                       / max(self.step_count, 1))
            info["time_to_first"] = (self.first_success_step / self.control_hz
                                     if reached else -1.0)

        self._update_memory(det)
        return self._obs(det), reward, terminated, truncated, info

    def close(self):
        self.camera.close()


def make_env(cfg, seed=None, rank=0):
    def _init():
        return SO101Approach3DOF(cfg, seed=None if seed is None else seed + rank)
    return _init