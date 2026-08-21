"""
reward_3dof.py — stage-1 reward, 3-DOF variant.

This is the 6-DOF reward with the two gripper-dependent pieces removed and
nothing else altered: same potential terms, same weights, same success
criteria, same dwell objective, same collision cost.

    REMOVED  w_grip          the gripper is a fixed joint; the term is constant
    REMOVED  grip_min/max    a success condition on a frozen joint is either
                             always true or always false
    KEPT     w_align         approach alignment is still a real quantity even
                             though the frozen elbow largely determines it

DWELL MODE
----------
The episode runs its full horizon. Holding the confirmed pose pays w_hold
every step, so return scales with time in pose and the optimal behaviour is
"arrive early, do not drift out". step_cost is 0.01/step against w_hold 0.5,
so holding is worth 50x the time it costs; a full 900-step hold is worth ~450
against a one-off arrival bonus of 20.

FIVE DESIGN DECISIONS
---------------------
1. SHAPING IS POTENTIAL-BASED.  r_shape = gamma * Phi(s') - Phi(s).
   Policy-invariant (Ng, Harada & Russell 1999), so dense shaping cannot
   create a spurious optimum. Terms that pull toward the goal live in Phi;
   terms that are genuine costs sit outside it, because they are MEANT to
   change the optimum.

2. THE ARRIVAL BONUS IS PAID ONCE PER EPISODE. Paid on every entry it would be
   farmable: leave the pose, re-enter, collect again.

3. VISIBILITY IS INSIDE THE POTENTIAL, NOT A PER-STEP BONUS. A per-step +1 for
   "cube in frame" is farmable — the best policy jitters the wrist to keep
   re-acquiring rather than progressing. Inside Phi it is banked once on the
   unseen->seen transition and refunded if sight is lost.

4. THE SMOOTHNESS TRIO IS THREE TERMS BECAUSE THEY CATCH DIFFERENT THINGS.
   w_smooth penalises action chatter (the command sequence), w_vel penalises
   joint speed (the resulting motion), w_accel penalises jerk. A policy can be
   smooth in commands and still whip the arm around, or move slowly while
   dithering the command every step. All three matter for the STS3215s.

5. THE COLLISION PENALTY IS NOT IN THE POTENTIAL. A potential term would
   refund the cost as the arm backs out, making a dip into the base free
   overall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])


# ----------------------------------------------------------------------
@dataclass
class RewardWeights:
    w_visible: float = 2.0
    w_center: float = 1.5
    w_xy: float = 6.0
    w_z: float = 4.0
    w_align: float = 2.0
    w_smooth: float = 0.15
    w_vel: float = 0.02
    w_accel: float = 0.005
    w_limit: float = 0.5
    w_collision: float = 1.0
    w_collision_depth: float = 40.0
    # Gripper below the ground-clearance line. Geometric, so unlike
    # w_collision it does not depend on the floor geom being named a
    # particular thing in the scene — it fires on height alone.
    w_ground: float = 3.0
    w_effort: float = 0.005
    step_cost: float = 0.01
    w_hold: float = 0.5
    # Paid every step the cube is VISIBLE, and deliberately OUTSIDE the
    # potential.
    #
    # Potential-based shaping is gamma*Phi(s') - Phi(s), which for a CONSTANT
    # Phi equals (gamma-1)*Phi: a cost of 0.01*Phi per step at gamma=0.99.
    # With the gated potential reaching 15.5 that is -0.155/step just for
    # keeping the cube centred, while looking away costs nothing. Measured in
    # a real run: detect_rate peaked at 0.80 by step 29k, then decayed to
    # 0.002 by 500k. The agent learned to see the cube and then correctly
    # unlearned it, because w_hold never paid — the pose was never reached.
    #
    # Inside the potential this term would telescope away and reintroduce the
    # same drift, so it sits outside. It is STATE-based rather than
    # transition-based, so the only way to farm it is to keep the cube in
    # frame, which is precisely the wanted behaviour. 0.0 disables it.
    w_seen_step: float = 0.25
    success_bonus: float = 20.0
    failure_penalty: float = 2.0
    # Multiply every privileged-pose term by visibility. See potential().
    # False reproduces the original ungated shaping exactly.
    gate_shaping_on_visibility: bool = True

    gamma: float = 0.99


@dataclass
class SuccessCriteria:
    center_tol: float = 0.40
    xy_tol: float = 0.020
    z_min: float = 0.045        # STANDOFF above the cube, not absolute height
    z_max: float = 0.075
    align_min: float = 0.94     # ~20 deg from straight down
    speed_max: float = 0.03
    hold_steps: int = 15


@dataclass
class Ctx:
    cube_pos: np.ndarray        # privileged, world frame
    pinch_pos: np.ndarray       # world frame
    pinch_vel: np.ndarray
    approach_axis: np.ndarray   # gripper -Z in world; want (0, 0, -1)
    detection: np.ndarray       # [u, v, size, visible]
    action: np.ndarray          # 3-dim, in [-1, 1]
    prev_action: np.ndarray
    qvel: np.ndarray            # trained joints
    prev_qvel: np.ndarray
    q_norm: np.ndarray          # trained joints mapped to [-1, 1] of range
    out_of_bounds: bool
    collision: bool = False
    collision_depth: float = 0.0
    ground_depth: float = 0.0   # metres the gripper is below clearance, >= 0


# ----------------------------------------------------------------------
class Stage1Reward:
    def __init__(self, weights: RewardWeights, criteria: SuccessCriteria):
        self.w = weights
        self.c = criteria
        self.reset()

    def reset(self):
        self._prev_phi = None
        self._hold = 0
        self._dwell = 0
        self._bonus_paid = False
        self._prev_oob = False

    @property
    def hold_counter(self) -> int:
        return self._hold

    @property
    def dwell_steps(self) -> int:
        return self._dwell

    # -- the potential --------------------------------------------------
    def potential(self, ctx: Ctx) -> float:
        w, c = self.w, self.c
        seen = ctx.detection[3] > 0.5

        d = ctx.pinch_pos - ctx.cube_pos
        dxy = float(np.linalg.norm(d[:2]))
        z_mid = 0.5 * (c.z_min + c.z_max)
        dz = abs(float(d[2]) - z_mid)
        align = float(np.dot(ctx.approach_axis, DOWN)
                      / (np.linalg.norm(ctx.approach_axis) + 1e-12))
        ce = min(float(np.linalg.norm(ctx.detection[:2])), 1.5)

        if not w.gate_shaping_on_visibility:
            # Original ungated form. Kept so a run can be reproduced exactly.
            phi = (w.w_visible * float(seen)
                   - w.w_xy * dxy - w.w_z * dz
                   - w.w_align * 0.5 * (1.0 - align))
            if seen:
                phi -= w.w_center * ce
            return phi

        # ------------------------------------------------------------------
        # GATED FORM — everything that depends on the true cube pose is
        # multiplied by visibility.
        #
        # The problem with the ungated version: the position terms read the
        # PRIVILEGED cube pose, so while the cube is off-camera two states
        # with byte-identical observations receive opposite shaping gradients
        # depending on which side the invisible cube happens to be. The
        # expected gradient is near zero, but the variance is large and it
        # does not shrink with more data, because it is conditioned on hidden
        # state the policy cannot see. The critic cannot fit that, and the
        # lowest-variance behaviour available to the actor is to stop moving —
        # which is precisely a policy that will not search.
        #
        # Gating leaves search driven by one clean, observable fact: getting
        # the cube into frame raises the potential. Everything else waits
        # until there is something to servo on.
        #
        # Still potential-based, so still policy-invariant: visibility is a
        # function of state, and Phi remains a function of state alone.
        #
        # Terms are written as REWARDS in [0, w] rather than penalties, and
        # bounded with tanh. That guarantees Phi(seen) >= Phi(unseen) = 0, so
        # acquiring the cube can never be punished. Written as penalties, a
        # distant sighting would give Phi = 2.0 - 6*0.3 - ... < 0 and the
        # agent would be rewarded for looking away again.
        # ------------------------------------------------------------------
        if not seen:
            return 0.0

        return (w.w_visible
                + w.w_xy * (1.0 - math.tanh(dxy / 0.15))
                + w.w_z * (1.0 - math.tanh(dz / 0.05))
                + w.w_align * 0.5 * (1.0 + align)
                + w.w_center * (1.0 - ce / 1.5))

    # -- success ---------------------------------------------------------
    def _conditions(self, ctx: Ctx) -> dict:
        c = self.c
        det = ctx.detection
        seen = det[3] > 0.5
        d = ctx.pinch_pos - ctx.cube_pos
        return {
            "detected": bool(seen),
            "centered": bool(seen and abs(det[0]) <= c.center_tol
                             and abs(det[1]) <= c.center_tol),
            "xy": bool(np.linalg.norm(d[:2]) <= c.xy_tol),
            "standoff": bool(c.z_min <= d[2] <= c.z_max),
            "align": bool(np.dot(ctx.approach_axis, DOWN) >= c.align_min),
            "slow": bool(np.linalg.norm(ctx.pinch_vel) <= c.speed_max),
        }

    # -- main ------------------------------------------------------------
    def __call__(self, ctx: Ctx) -> tuple[float, bool, dict]:
        """Returns (reward, in_pose, components).

        `in_pose` is True on EVERY step the confirmed pose is held. In dwell
        mode the env uses it to timestamp first arrival and ignores it for
        termination; in terminate_on_success mode the env ends on it.
        """
        w = self.w
        phi = self.potential(ctx)
        shaped = 0.0 if self._prev_phi is None else (w.gamma * phi - self._prev_phi)
        self._prev_phi = phi

        smooth = -w.w_smooth * float(np.sum((ctx.action - ctx.prev_action) ** 2))
        vel = -w.w_vel * float(np.sum(ctx.qvel ** 2))
        accel = -w.w_accel * float(np.sum((ctx.qvel - ctx.prev_qvel) ** 2))
        effort = -w.w_effort * float(np.sum(ctx.action ** 2))

        # Soft barrier: zero through the middle of the range, quadratic in the
        # outer 10%. A hard clip teaches nothing — the agent keeps commanding
        # into the stop and the servo grinds.
        over = np.clip(np.abs(ctx.q_norm) - 0.9, 0.0, None)
        limit = -w.w_limit * float(np.sum((over / 0.1) ** 2))

        collide = 0.0
        if ctx.collision:
            collide = -(w.w_collision + w.w_collision_depth * ctx.collision_depth)

        # Scraping the floor passes the height check in sim and strips a gear
        # on hardware. Linear in depth rather than flat: skimming the surface
        # should cost less than driving into it, and a flat penalty gives no
        # gradient telling the policy which way is out.
        ground = -w.w_ground * ctx.ground_depth / 0.01 if ctx.ground_depth > 0 else 0.0

        # Must exceed the (1-gamma)*Phi drift, or keeping the cube in view is
        # still net negative. Max drift is 0.01*15.5 = 0.155/step.
        seen_pay = w.w_seen_step if ctx.detection[3] > 0.5 else 0.0

        r = (shaped + smooth + vel + accel + effort + limit + collide + ground
             + seen_pay - w.step_cost)

        cond = self._conditions(ctx)
        self._hold = self._hold + 1 if all(cond.values()) else 0
        in_pose = self._hold >= self.c.hold_steps

        hold_pay = 0.0
        if in_pose:
            self._dwell += 1
            hold_pay = w.w_hold
            if not self._bonus_paid:
                hold_pay += w.success_bonus     # once per episode, decision 2
                self._bonus_paid = True
        r += hold_pay

        # EDGE-triggered, not per-step. In dwell mode leaving the workspace no
        # longer ends the episode, so a per-step charge would reach -1800 on a
        # 900-step horizon against a +450 objective, and the agent would learn
        # to sit at home rather than risk approaching.
        oob = 0.0
        if ctx.out_of_bounds and not self._prev_oob:
            oob = -w.failure_penalty
            r += oob
        self._prev_oob = bool(ctx.out_of_bounds)

        comps = {"shaped": shaped, "smooth": smooth, "vel": vel, "accel": accel,
                 "effort": effort, "limit": limit, "collide": collide,
                 "ground": ground, "seen_pay": seen_pay,
                 "hold_pay": hold_pay, "oob": oob, "phi": phi,
                 "hold": float(self._hold), "dwell": float(self._dwell),
                 **{f"ok_{k}": float(v) for k, v in cond.items()}}
        return float(r), bool(in_pose), comps