#!/usr/bin/env python3
"""verify_setup.py — check every file is present and consistent before training.

    python verify_setup.py

Catches the failure modes that have actually happened in this project: a file
downloaded at the wrong version, a config key with no matching dataclass
field, joint lists whose lengths disagree, and software GL rendering
masquerading as hardware.
"""
import ast, importlib.util, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ok, fail = [], []
def check(cond, label, hint=""):
    (ok if cond else fail).append((label, hint))
    print(f"  [{'ok' if cond else 'FAIL'}] {label}")
    if not cond and hint:
        for l in hint.splitlines(): print(f"         {l}")

print("1. files present")
need = ["env_3dof.py","reward_3dof.py","rate.py","train_3dof.py","train_supervised.py",
        "sanity_check.py","evaluate.py","visualize_policy.py","view_env.py",
        "set_angles.py","check_gl.py","vision/cv_detector.py","vision/__init__.py",
        "tools/benchmark.py","tools/fit_spawn.py","tools/tune_hsv.py",
        "tools/setup_fixed_angles.py","configs/stage1_3dof.yaml"]
for f in need:
    check((HERE/f).exists(), f, "missing — re-download")
check((HERE/"mujoco").is_dir() and any((HERE/"mujoco").glob("*.xml")),
      "mujoco/*.xml", "copy train_scene.xml and its model files in")

print("\n2. config <-> dataclass")
import yaml
cfg = yaml.safe_load((HERE/"configs/stage1_3dof.yaml").read_text())
def fields(src, cls):
    t = ast.parse(Path(src).read_text())
    return {x.target.id for k in t.body if isinstance(k, ast.ClassDef) and k.name == cls
            for x in k.body if isinstance(x, ast.AnnAssign)}
for sec, src, cls in [(("reward","weights"), "reward_3dof.py", "RewardWeights"),
                      (("reward","success"), "reward_3dof.py", "SuccessCriteria"),
                      (("detector",), "vision/cv_detector.py", "DetectorConfig")]:
    d = cfg
    for k in sec: d = d[k]
    f = fields(HERE/src, cls)
    extra, miss = set(d) - f, f - set(d)
    check(not extra and not miss, ".".join(sec) + f" <-> {cls}",
          f"config has {extra or '{}'}, dataclass wants {miss or '{}'}\n"
          f"you have mismatched versions of the config and {src}")

print("\n3. joint lists agree")
e = cfg["env"]
n = len(e.get("trained_joints", ["a","b","c"]))
check(len(e["home_qpos"]) == n, f"home_qpos has {n} entries",
      f"trained_joints has {n}, home_qpos has {len(e['home_qpos'])}")
check(len(e["max_delta"]) == n, f"max_delta has {n} entries",
      f"trained_joints has {n}, max_delta has {len(e['max_delta'])}")
print(f"         trained: {e.get('trained_joints')}  -> action dim {n}, "
      f"obs dim {3*n+5+(3 if e.get('last_seen_memory') else 0)}")

print("\n4. rates and rollout")
check(abs(cfg["reward"]["weights"]["gamma"] - cfg["train"]["gamma"]) < 1e-9,
      "reward gamma == train gamma", "potential shaping is not invariant otherwise")
roll = cfg["train"]["n_steps"] * cfg["train"]["n_envs"]
ep = int(e["episode_seconds"] * e["control_hz"])
check(roll % cfg["train"]["batch_size"] == 0,
      f"rollout {roll} divisible by batch {cfg['train']['batch_size']}")
check(roll/ep >= 3, f"rollout = {roll/ep:.1f} episodes",
      "under ~3 episodes per update makes advantages noisy; raise n_steps")

print("\n5. fixed angles")
fa = HERE / e["fixed_angles"]
if fa.exists():
    got = yaml.safe_load(fa.read_text())["fixed_joints"]
    ARM = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"]
    need_fixed = [j for j in ARM if j not in e.get("trained_joints", [])]
    check(all(j in got for j in need_fixed), f"fixed_angles has {need_fixed}",
          f"file has {list(got)}")
    print("         " + "  ".join(f"{k}={v:+.3f}" for k, v in got.items()))
else:
    check(False, str(fa), "run set_angles.py")

print("\n6. GL backend")
os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYOPENGL_PLATFORM","egl")
import glob
icd = [Path(p).name for p in glob.glob("/usr/share/glvnd/egl_vendor.d/*.json")]
hw = any("nvidia" in i for i in icd)
check(bool(icd), f"EGL vendor ICDs {icd}", "no EGL driver registered")
check(hw, "hardware EGL (nvidia ICD present)",
      f"only {icd} -> Mesa SOFTWARE rendering on the CPU.\n"
      f"Measured 4.3 steps/s per core vs 100 with a GPU: a 23x penalty.\n"
      f"You are on a CPU-only instance; move to one with a GPU (AWS g5.*).")

print("\n" + "="*66)
if fail:
    print(f"{len(fail)} PROBLEM(S) — fix before training")
    sys.exit(1)
print("all checks passed — safe to train")