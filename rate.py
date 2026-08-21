"""
rate.py — derive every rate-dependent setting from ONE knob: env.control_hz.

WHY THIS EXISTS
---------------
Changing the control rate by hand means editing seven values that are counted
in STEPS rather than seconds, and forgetting any one of them silently changes
the task instead of the rate. Drop from 30 Hz to 5 Hz and leave max_delta
alone and the arm crawls at a sixth of its speed; leave gamma alone and the
agent's planning horizon stretches from 3.3 s to 20 s.

So: set env.control_hz. Everything else is derived here.

THE RULE
--------
Every number in the config is authored at env.reference_hz (30 Hz — the rate
all the existing values were tuned at). Let r = reference_hz / control_hz.

    PER-STEP quantities scale by r      because there are now r times fewer
                                        steps per second, so each must carry
                                        r times as much
    PER-EVENT quantities do not         they fire once, whatever the rate
    PHYSICAL quantities do not          metres, radians and seconds do not
                                        care how often you sample

    scaled by r         max_delta, w_hold, step_cost, w_smooth, w_vel,
                        w_accel, w_effort, w_limit
    scaled by 1/r       hold_steps, action_delay_min/max, total_timesteps,
                        checkpoint_every
    exponentiated       gamma -> gamma**r, action_filter -> filter**r
                        (both are per-step decays; raising to r preserves the
                        time constant rather than the per-step coefficient)
    untouched           success_bonus, failure_penalty, every tolerance,
                        every distance, episode_seconds

GAMMA IS THE ONE PEOPLE MISS. gamma=0.99 is per step, so the effective horizon
is 1/(1-gamma) = 100 steps. At 30 Hz that is 3.3 s. At 5 Hz the same 0.99
becomes 20 s and you are training a completely different agent without having
changed a line of reward code. 0.99**6 = 0.941 keeps the horizon at 3.3 s.

IDEMPOTENT. resolve() marks the dict and returns early if called twice, so it
is safe for env, train and the tools all to call it on the same config.
"""

from __future__ import annotations

import math

REFERENCE_HZ = 30.0
_MARK = "_rate_resolved"


def resolve(cfg: dict, verbose: bool = False) -> dict:
    """Resolve cfg in place for cfg['env']['control_hz']. Returns cfg."""
    if cfg.get(_MARK):
        return cfg

    env = cfg["env"]
    physics_hz = float(env.get("physics_hz", 600.0))
    ref = float(env.get("reference_hz", REFERENCE_HZ))

    # Back-compat: a config that still specifies frame_skip and no control_hz
    # keeps working, and resolves to exactly what it did before.
    if "control_hz" not in env:
        env["control_hz"] = physics_hz / float(env["frame_skip"])
    hz = float(env["control_hz"])

    fs = physics_hz / hz
    if abs(fs - round(fs)) > 1e-9:
        raise ValueError(
            f"control_hz={hz} does not divide physics_hz={physics_hz:.0f} "
            f"evenly (frame_skip would be {fs:.3f}). Pick a rate from "
            f"{sorted({physics_hz/k for k in range(1, 201) if physics_hz % k == 0 and physics_hz/k <= 120})}"
            .replace(".0,", ",").replace(".0]", "]"))
    env["frame_skip"] = int(round(fs))

    r = ref / hz
    if abs(r - 1.0) < 1e-12:
        cfg[_MARK] = True
        if verbose:
            _report(cfg, hz, r)
        return cfg

    w = cfg["reward"]["weights"]
    sc = cfg["reward"]["success"]
    rnd = cfg.get("randomization", {})
    tr = cfg["train"]

    # --- per-step magnitudes: scale UP as steps get rarer ---
    env["max_delta"] = [float(v) * r for v in env["max_delta"]]
    for k in ("w_hold", "step_cost", "w_smooth", "w_vel", "w_accel",
              "w_effort", "w_limit"):
        if k in w:
            w[k] = float(w[k]) * r

    # --- per-step counts: scale DOWN ---
    # ceil, not round: hold_steps must cover at least the intended duration,
    # and it must never reach 0.
    sc["hold_steps"] = max(1, math.ceil(float(sc["hold_steps"]) / r))
    for k in ("action_delay_min", "action_delay_max"):
        if k in rnd:
            rnd[k] = int(round(float(rnd[k]) / r))
    if rnd.get("action_delay_max", 0) < rnd.get("action_delay_min", 0):
        rnd["action_delay_max"] = rnd["action_delay_min"]

    # --- per-step decays: exponentiate to preserve the TIME constant ---
    g = float(w["gamma"]) ** r
    w["gamma"] = g
    tr["gamma"] = g
    if float(env.get("action_filter", 0.0)) > 0.0:
        env["action_filter"] = float(env["action_filter"]) ** r

    # --- budget: same number of episodes, not the same number of steps ---
    tr["total_timesteps"] = int(round(float(tr["total_timesteps"]) / r))
    if "checkpoint_every" in tr:
        tr["checkpoint_every"] = max(1000, int(round(
            float(tr["checkpoint_every"]) / r)))

    cfg[_MARK] = True
    if verbose:
        _report(cfg, hz, r)
    return cfg


def _report(cfg, hz, r):
    env, w, sc, tr = (cfg["env"], cfg["reward"]["weights"],
                      cfg["reward"]["success"], cfg["train"])
    rnd = cfg.get("randomization", {})
    steps = int(round(env["episode_seconds"] * hz))
    print(f"control rate      {hz:g} Hz   (frame_skip {env['frame_skip']}, "
          f"scale r = {r:g} vs {env.get('reference_hz', REFERENCE_HZ):g} Hz)")
    print(f"  episode         {env['episode_seconds']:g} s = {steps} steps")
    print(f"  max_delta       {[round(v, 4) for v in env['max_delta']]} rad/step"
          f"  = {env['max_delta'][0]*hz:.2f} rad/s")
    print(f"  action_filter   {env.get('action_filter', 0.0):.4f}")
    print(f"  action_delay    {rnd.get('action_delay_min', 0)}"
          f"-{rnd.get('action_delay_max', 0)} steps "
          f"= 0-{1000*rnd.get('action_delay_max', 0)/hz:.0f} ms")
    print(f"  hold_steps      {sc['hold_steps']} = "
          f"{sc['hold_steps']/hz:.2f} s")
    print(f"  gamma           {w['gamma']:.4f}  "
          f"(horizon {1/(1-w['gamma'])/hz:.1f} s)")
    print(f"  w_hold          {w['w_hold']:.3f}/step = "
          f"{w['w_hold']*hz:.1f}/s;  full hold = "
          f"{w['w_hold']*steps:.0f} vs bonus {w['success_bonus']:g}")
    print(f"  step_cost       {w['step_cost']:.4f}/step")
    print(f"  total_timesteps {tr['total_timesteps']:,} = "
          f"{tr['total_timesteps']/max(steps,1):,.0f} episodes")