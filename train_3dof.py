#!/usr/bin/env python3
"""
train_3dof.py — PPO on the 3-DOF stage-1 task.

    python train_3dof.py --config configs/stage1_3dof.yaml

Reads the `train:` block of the config unchanged from the 6-DOF project:
normalize_obs, normalize_reward, clip_obs, checkpoint_every, log_dir,
snapshot_dir, save_success_states.

SNAPSHOT LAYOUT (unchanged, so evaluate.py and visualize_policy.py find things
where they always did)

    policy/snapshots/<run_name>/<YYYYmmdd-HHMMSS>/
        best.pt                  best model by eval return  <- USE THIS
        vecnorm_best.pkl         its normaliser, saved at the same moment
        final.pt / vecnorm_final.pkl
        ckpt_<steps>.pt / vecnorm_ckpt_<steps>.pkl
        config.yaml              resolved config
        fixed_angles.yaml        the frozen angles this policy was trained with
    policy/snapshots/<run_name>/latest -> the newest timestamped directory

Set train.run_timestamp: false to go back to one flat directory per run_name.

WHICH CHECKPOINT TO LOAD
    best.pt          evaluating or demoing. Highest eval return.
    ckpt_<steps>.pt  diagnosing "it got worse after N steps", or resuming.
    final.pt         resuming; it is the last state, not the best one.

    Always pass the .pt — the matching vecnorm_*.pkl is found automatically
    from the filename, and it is REQUIRED because normalize_obs is true.

NORMALISER STATS ARE PART OF THE POLICY. normalize_obs is true, so a checkpoint
loaded without its vecnorm_*.pkl sees garbage inputs and scores like a random
policy. Every save writes both files together for that reason.

THINGS THAT WILL BITE YOU
-------------------------
1. MUJOCO_GL. Unset, MuJoCo tries GLFW and every subprocess worker fights over
   a display that is not there. This script sets egl if you have not.
2. gamma mismatch. reward.weights.gamma must equal train.gamma or the
   potential-based shaping stops being policy-invariant. Asserted below.
3. In dwell mode watch z/dwell_frac, not the binary success rate — the latter
   saturates at 1.0 once the policy can touch the pose at all.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# PyOpenGL selects its backend from PYOPENGL_PLATFORM, NOT from MUJOCO_GL.
# On a headless box with no display it otherwise resolves to None and fails
# with "'NoneType' object has no attribute 'eglQueryString'" deep inside the
# mujoco import. Setting both keeps them consistent.
os.environ.setdefault("PYOPENGL_PLATFORM",
                      os.environ.get("MUJOCO_GL", "egl"))
# One BLAS thread per process. With SubprocVecEnv, every worker otherwise
# spawns a full thread pool and they fight over the same cores; the policy
# update is small enough that it loses nothing by being single-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                EvalCallback)
from stable_baselines3.common.vec_env import (SubprocVecEnv, VecMonitor,
                                              VecNormalize)

from env_3dof import SO101Approach3DOF
from rate import resolve as resolve_rate

HERE = Path(__file__).resolve().parent


class ComponentLogger(BaseCallback):
    """Log the reward breakdown and the dwell metrics. Without this, a run
    that goes wrong tells you only that the return dropped, not which term."""

    def _on_step(self) -> bool:
        acc = {}
        for info in self.locals.get("infos", []):
            for k, v in info.items():
                if k.startswith("rc/"):
                    acc.setdefault(k, []).append(v)
            if "detected" in info:
                acc.setdefault("z/detect_rate", []).append(float(info["detected"]))
            if "collision" in info:
                acc.setdefault("z/collision_rate", []).append(
                    float(info["collision"]))
            if "episode_success" in info:
                acc.setdefault("z/episode_success", []).append(
                    info["episode_success"])
                # THE headline metric in dwell mode.
                acc.setdefault("z/dwell_frac", []).append(info["dwell_frac"])
                if info.get("time_to_first", -1.0) >= 0:
                    acc.setdefault("z/time_to_first_s", []).append(
                        info["time_to_first"])
        for k, v in acc.items():
            self.logger.record_mean(k, float(np.mean(v)))
        return True


class SnapshotCallback(BaseCallback):
    """Periodic checkpoint that saves the normaliser alongside the weights."""

    def __init__(self, every: int, out: Path, venv: VecNormalize,
                 start: int = 0):
        super().__init__()
        self.every, self.out, self.venv = every, out, venv
        # On a resume num_timesteps continues from the old run, so the first
        # checkpoint must be `every` steps from THERE, not from zero.
        self._next = start + every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            tag = f"ckpt_{self.num_timesteps}"
            self.model.save(str(self.out / f"{tag}.pt"))
            self.venv.save(str(self.out / f"vecnorm_{tag}.pkl"))
        return True


class BestSnapshot(EvalCallback):
    """EvalCallback that writes best.pt AND its normaliser together.

    Stock EvalCallback writes best_model.zip and nothing else. Two problems:
    the tools resolve statistics as vecnorm_<stem>.pkl, so best_model.zip
    would need vecnorm_best_model.pkl; and saving the normaliser at the end of
    training captures the statistics as they were THEN, not as they were when
    this checkpoint was the best one. With normalize_obs=true a mismatched
    normaliser makes a good policy score like a random one, silently.

    So: save both, at the moment of improvement, under matching names.
    """

    def __init__(self, *a, venv=None, out: Path = None, **kw):
        super().__init__(*a, **kw)
        self.venv, self.out = venv, out
        self._best = -float("inf")

    def _on_step(self) -> bool:
        keep = super()._on_step()
        if self.best_mean_reward > self._best:
            self._best = self.best_mean_reward
            self.model.save(str(self.out / "best.pt"))
            self.venv.save(str(self.out / "vecnorm_best.pkl"))
            if self.verbose:
                print(f"  new best {self.best_mean_reward:.1f} -> best.pt "
                      f"+ vecnorm_best.pkl")
        return keep


def build(cfg, seed, n_envs, training=True, vecnorm=None):
    """Build the vec env. `vecnorm` restores saved statistics for a resume.

    Restoring them is not optional. normalize_obs is true, so the running mean
    and variance ARE part of the policy: reload the weights with fresh
    statistics and the network sees differently-scaled inputs than it was
    trained on. It does not error, it just quietly performs like a random
    policy, and you lose the run before noticing.
    """
    t = cfg["train"]

    def mk(rank):
        # No per-env Monitor: VecMonitor below records the same episode
        # statistics and would overwrite these anyway (SB3 warns about it).
        def _init():
            return SO101Approach3DOF(cfg, seed=seed + rank)
        return _init

    venv = VecMonitor(SubprocVecEnv([mk(i) for i in range(n_envs)]))

    if vecnorm is not None:
        vn = VecNormalize.load(str(vecnorm), venv)
        vn.training = training
        vn.norm_reward = bool(t["normalize_reward"]) and training
        return vn

    return VecNormalize(venv,
                        norm_obs=bool(t["normalize_obs"]),
                        norm_reward=bool(t["normalize_reward"]) and training,
                        clip_obs=float(t["clip_obs"]),
                        training=training,
                        gamma=float(t["gamma"]))


def resolve_resume(spec: str, root: Path, run_name: str) -> tuple[Path, Path]:
    """Turn --resume into (weights, statistics), or exit explaining why not.

    Accepts a path to a .pt, or the word `latest` to pick the newest run's
    final.pt.
    """
    if spec in ("latest", "auto"):
        runs = root / run_name
        cands = []
        if runs.exists():
            for d in runs.iterdir():
                # Skip `latest` — it is a symlink INTO this same directory, so
                # following it makes the resume tag "latest", and the run then
                # tries to point the latest symlink at itself. That fails with
                # ELOOP when anything is written into it.
                if d.is_symlink() or not d.is_dir():
                    continue
                for w in d.glob("*.pt"):
                    if not (d / f"vecnorm_{w.stem}.pkl").exists():
                        continue          # unusable without its statistics
                    # Rank by trained steps, not mtime. A crash leaves the
                    # newest FILE as whatever was mid-write; the highest step
                    # count is the furthest the run actually got. ckpt_<n>.pt
                    # carries it in the name; final.pt is written on a clean
                    # exit so it is at least as far along as any ckpt.
                    m = re.match(r"ckpt_(\d+)$", w.stem)
                    if m:
                        rank = int(m.group(1))
                    elif w.stem == "final":
                        rank = float("inf")
                    else:
                        continue          # best.pt: fewest steps, not a resume
                    cands.append((rank, d.stat().st_mtime, w))
        if not cands:
            raise SystemExit(
                f"no resumable checkpoint under {runs}\n"
                "(needs a ckpt_*.pt or final.pt WITH its vecnorm_*.pkl)")
        cands.sort(key=lambda c: (c[0], c[1]))
        weights = cands[-1][2]
        print(f"--resume {spec} -> {weights}")
    else:
        weights = Path(spec)
        if not weights.exists():
            raise SystemExit(f"checkpoint not found: {weights}")
        # An explicit path through `latest` has the same self-reference
        # problem, so resolve it to the real timestamped directory.
        weights = weights.resolve()

    stats = weights.parent / f"vecnorm_{weights.stem}.pkl"
    if not stats.exists():
        raise SystemExit(
            f"statistics missing: {stats}\n"
            "normalize_obs is true, so resuming without them would feed the "
            "policy differently-scaled observations than it was trained on. "
            "Every checkpoint is written together with its vecnorm_*.pkl — "
            "find the pair, or start fresh.")
    return weights, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--new-dir", action="store_true",
                    help="on resume, start a new timestamped directory "
                         "instead of continuing the old one. Branches the "
                         "TensorBoard curve.")
    ap.add_argument("--resume", default=None,
                    help="path to a .pt checkpoint, or 'latest' for the newest "
                         "run's final.pt. Its vecnorm_*.pkl is loaded too.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = HERE / cfg_path
    cfg = resolve_rate(yaml.safe_load(cfg_path.read_text()), verbose=True)
    t = cfg["train"]
    if args.n_envs:
        t["n_envs"] = args.n_envs

    assert abs(cfg["reward"]["weights"]["gamma"] - t["gamma"]) < 1e-9, (
        "reward.weights.gamma must equal train.gamma, otherwise the "
        "potential-based shaping is no longer policy-invariant.")

    rollout = int(t["n_steps"]) * int(t["n_envs"])
    if rollout % int(t["batch_size"]) != 0:
        for cand in (512, 256, 128, 64, 32):
            if rollout % cand == 0:
                print(f"batch_size {t['batch_size']} does not divide a "
                      f"{rollout}-sample rollout; using {cand}.")
                t["batch_size"] = cand
                break

    resume_w = resume_s = None
    if args.resume:
        resume_w, resume_s = resolve_resume(
            args.resume, HERE / t["snapshot_dir"], cfg["run_name"])
        prev_cfg = resume_w.parent / "config.yaml"
        if prev_cfg.exists():
            old = yaml.safe_load(prev_cfg.read_text())
            for path in (("env", "control_hz"), ("env", "episode_seconds"),
                         ("train", "normalize_obs"), ("env", "fixed_angles")):
                a_, b_ = old, cfg
                for k in path:
                    a_, b_ = (a_ or {}).get(k), (b_ or {}).get(k)
                if a_ != b_:
                    print(f"  WARNING: {'.'.join(path)} changed since that "
                          f"checkpoint: {a_!r} -> {b_!r}")

    # A RESUME CONTINUES THE SAME RUN DIRECTORY, it does not open a new one.
    #
    # A new directory means a new TensorBoard event file, which TensorBoard
    # reads as a separate run: the curve breaks in two and you cannot see the
    # whole training at once. Writing back into the original directory with
    # reset_num_timesteps=False makes SB3 reuse the existing PPO_1 folder and
    # keep counting, so the graph is continuous.
    #
    # A fresh run still gets its own timestamp, so two independent runs never
    # overwrite each other's best.pt. --new-dir forces one on a resume.
    if resume_w and not args.new_dir:
        tag = resume_w.parent.name
        print(f"continuing in {tag} (TensorBoard sees one unbroken run; "
              f"--new-dir to branch instead)")
    else:
        tag = (datetime.now().strftime("%Y%m%d-%H%M%S")
               if t.get("run_timestamp", True) else "run")
    snap_dir = HERE / t["snapshot_dir"] / cfg["run_name"] / tag
    log_dir = HERE / t["log_dir"] / cfg["run_name"] / tag
    snap_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for parent, target in ((snap_dir.parent, snap_dir), (log_dir.parent, log_dir)):
        link = parent / "latest"
        if target.name == "latest":
            continue                          # never link a directory to itself
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target.name)      # relative: survives a move
        except OSError:
            pass                              # filesystems without symlinks
    # On a resume the original config.yaml stays put — evaluate.py reads it
    # to rebuild the env, and the run that produced these weights is the one
    # worth recording. Later configs land beside it, numbered.
    dest = snap_dir / "config.yaml"
    if dest.exists() and resume_w:
        n = len(list(snap_dir.glob("config.resumed*.yaml"))) + 1
        dest = snap_dir / f"config.resumed{n}.yaml"
    shutil.copy(cfg_path, dest)
    fa = HERE / cfg["env"]["fixed_angles"]
    if fa.exists():
        shutil.copy(fa, snap_dir / "fixed_angles.yaml")

    seed = int(t["seed"])
    train_env = build(cfg, seed, int(t["n_envs"]), training=True,
                      vecnorm=resume_s)
    eval_env = build(cfg, seed + 1000, 2, training=False, vecnorm=resume_s)

    cbs = CallbackList([
        ComponentLogger(),
        SnapshotCallback(int(t["checkpoint_every"]), snap_dir, train_env,
                         start=0),   # patched below once the model exists
        BestSnapshot(eval_env,
                     venv=train_env, out=snap_dir,
                     best_model_save_path=None,   # we write best.pt ourselves
                     log_path=str(log_dir / "eval"),
                     # eval_freq is PER WORKER: SB3 counts callback steps in
                     # the vec env, so //n_envs makes it a total-timestep
                     # figure and eval_every means what it says.
                     eval_freq=max(1, int(t.get("eval_every",
                                                t["checkpoint_every"]))
                                   // int(t["n_envs"])),
                     n_eval_episodes=int(t.get("n_eval_episodes", 15)),
                     deterministic=True, verbose=1),
    ])

    if resume_w:
        model = PPO.load(str(resume_w), env=train_env,
                         device=t.get("device", "cpu"))
        print(f"resumed at {model.num_timesteps:,} steps "
              f"(statistics from {resume_s.name})")

        # PPO.load restores hyperparameters FROM THE ZIP, so edits to the
        # train: block are silently ignored on resume — you change ent_coef,
        # nothing happens, and the run looks identical to before. Re-apply
        # them here, and say which ones actually changed.
        #
        # learning_rate needs lr_schedule too: SB3 calls the schedule every
        # update and ignores the scalar, so setting only the attribute has no
        # effect whatsoever.
        lr = float(t["learning_rate"])
        changed = []
        for attr, want in [("ent_coef", float(t["ent_coef"])),
                           ("vf_coef", float(t["vf_coef"])),
                           ("clip_range", float(t["clip_range"])),
                           ("gamma", float(t["gamma"])),
                           ("gae_lambda", float(t["gae_lambda"])),
                           ("max_grad_norm", float(t["max_grad_norm"])),
                           ("n_epochs", int(t["n_epochs"])),
                           ("batch_size", int(t["batch_size"]))]:
            have = getattr(model, attr, None)
            if callable(have):                       # clip_range is a schedule
                have = have(1.0)
            if have is not None and abs(float(have) - float(want)) > 1e-12:
                changed.append(f"{attr} {have:g} -> {want:g}")
            if attr == "clip_range":
                model.clip_range = lambda _, v=want: v
            else:
                setattr(model, attr, want)

        old_lr = model.lr_schedule(1.0)
        if abs(old_lr - lr) > 1e-12:
            changed.append(f"learning_rate {old_lr:g} -> {lr:g}")
        model.learning_rate = lr
        model.lr_schedule = lambda _: lr
        for g in model.policy.optimizer.param_groups:
            g["lr"] = lr

        if changed:
            print("  re-applied from config: " + "; ".join(changed))
        else:
            print("  hyperparameters unchanged from the checkpoint")
    else:
        model = PPO("MlpPolicy", train_env,
                    learning_rate=float(t["learning_rate"]),
                    n_steps=int(t["n_steps"]),
                    batch_size=int(t["batch_size"]),
                    n_epochs=int(t["n_epochs"]),
                    gamma=float(t["gamma"]),
                    gae_lambda=float(t["gae_lambda"]),
                    clip_range=float(t["clip_range"]),
                    ent_coef=float(t["ent_coef"]),
                    vf_coef=float(t["vf_coef"]),
                    max_grad_norm=float(t["max_grad_norm"]),
                    policy_kwargs=dict(net_arch=list(t["net_arch"])),
                    tensorboard_log=str(log_dir / "tb"),
                    # A 256x256 MLP is too small to fill a GPU; the transfers
                    # cost more than the matmuls save, and SB3 warns about it.
                    # The GPU is better left free here — with 8 renderers the
                    # bottleneck is MuJoCo, not the policy update.
                    device=t.get("device", "cpu"),
                    seed=seed, verbose=1)

    # progress_bar needs tqdm+rich. It is cosmetic, so a missing optional
    # dependency must not abort a multi-hour run.
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401
        bar = True
    except ImportError:
        bar = False
        print("tqdm/rich not installed — running without the progress bar "
              "(pip install tqdm rich to get it back).")

    # SB3 adds num_timesteps to the target when reset_num_timesteps=False, so
    # pass the REMAINING budget to land on the configured total rather than
    # training the full amount all over again.
    done = model.num_timesteps
    remaining = max(0, int(t["total_timesteps"]) - done)
    for cb in cbs.callbacks:
        if isinstance(cb, SnapshotCallback):
            cb._next = done + cb.every
    if done:
        print(f"{done:,} steps already done, {remaining:,} to go "
              f"(target {int(t['total_timesteps']):,})")
    if remaining == 0:
        print("budget already reached; raise train.total_timesteps to continue")

    try:
        model.learn(total_timesteps=remaining, callback=cbs,
                    progress_bar=bar, reset_num_timesteps=not bool(resume_w))
    finally:
        # Ctrl-C still leaves you a usable pair of files.
        model.save(str(snap_dir / "final.pt"))
        train_env.save(str(snap_dir / "vecnorm_final.pkl"))
        train_env.close()
        eval_env.close()
    print(f"done -> {snap_dir}")


if __name__ == "__main__":
    main()