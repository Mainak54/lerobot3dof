#!/usr/bin/env python3
"""
train_supervised.py — run training to completion, restarting after crashes.

    python train_supervised.py
    python train_supervised.py --config configs/stage1_3dof.yaml
    python train_supervised.py --max-restarts 20 --fresh

Launches train_3dof.py as a subprocess and relaunches it with `--resume auto`
whenever it dies unexpectedly, until the configured total_timesteps is reached
or you stop it. Ctrl-C stops everything and does NOT restart.

WHY A SEPARATE PROCESS RATHER THAN A try/except INSIDE TRAINING
----------------------------------------------------------------
The failures worth surviving are the ones a try/except cannot catch. A MuJoCo
or GL segfault takes the interpreter down with no traceback and no chance to
run a finally block. The OOM killer sends SIGKILL. A wedged EGL context hangs
a worker forever. None of that is recoverable in-process — only a supervisor
outside it can notice and start again.

That also dictates which checkpoint to resume from. train_3dof.py writes
final.pt in a finally block, so a clean exit or Ctrl-C leaves one behind — but
a segfault does not. `--resume auto` therefore ranks candidates by TRAINED
STEPS parsed from ckpt_<n>.pt, not by file mtime, and skips any checkpoint
whose vecnorm_*.pkl is missing. After a crash you lose at most
checkpoint_every steps of progress.

CRASH-LOOP PROTECTION
---------------------
A restart that makes no progress is not a recovery, it is a loop. The
supervisor tracks the highest checkpoint step it has seen; restarts that fail
to advance it count as consecutive failures, and after --max-restarts of those
in a row it gives up rather than burning the night rewriting the same
checkpoint. A restart that DID progress resets the counter, so a run that
crashes every few hours can continue indefinitely.

EXIT CODES
    0   budget reached
    1   gave up after repeated no-progress failures
    130 you pressed Ctrl-C
"""

from __future__ import annotations

import argparse
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def highest_step(snap_root: Path) -> int:
    """Furthest trained step across every run directory, 0 if none.

    Only counts checkpoints that have their statistics file: one without it
    cannot be resumed, so it does not represent recoverable progress.
    """
    best = 0
    if not snap_root.exists():
        return 0
    for d in snap_root.iterdir():
        if not d.is_dir():
            continue
        for w in d.glob("ckpt_*.pt"):
            m = re.match(r"ckpt_(\d+)$", w.stem)
            if m and (d / f"vecnorm_{w.stem}.pkl").exists():
                best = max(best, int(m.group(1)))
    return best


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--max-restarts", type=int, default=10,
                    help="consecutive no-progress restarts before giving up")
    ap.add_argument("--backoff", type=float, default=15.0,
                    help="seconds to wait after a crash; doubles each "
                         "consecutive failure, capped at 5 minutes")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and start from scratch")
    ap.add_argument("--python", default=sys.executable)
    a, extra = ap.parse_known_args()

    cfg_path = HERE / a.config if not Path(a.config).is_absolute() else Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    t = cfg["train"]
    snap_root = HERE / t["snapshot_dir"] / cfg["run_name"]
    target = int(t["total_timesteps"])

    print("=" * 72)
    print(f"SUPERVISOR   target {target:,} steps   config {cfg_path.name}")
    print(f"             snapshots {snap_root}")
    print(f"             Ctrl-C stops for good; crashes restart from the last "
          f"checkpoint")
    print("=" * 72)

    attempt = 0
    consecutive = 0
    seen = 0 if a.fresh else highest_step(snap_root)
    if seen:
        print(f"[{stamp()}] found existing progress at {seen:,} steps")

    while True:
        attempt += 1
        cmd = [a.python, str(HERE / "train_3dof.py"), "--config", str(cfg_path)]
        if not (a.fresh and attempt == 1) and highest_step(snap_root) > 0:
            cmd += ["--resume", "auto"]
        cmd += extra

        print(f"\n[{stamp()}] attempt {attempt}: {' '.join(cmd[1:])}\n")
        t0 = time.time()
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            # Ctrl-C reaches the child too; give it a moment to write final.pt.
            print(f"\n[{stamp()}] interrupted — not restarting")
            time.sleep(2)
            return 130

        mins = (time.time() - t0) / 60
        now = highest_step(snap_root)

        if rc == 0:
            print(f"\n[{stamp()}] training exited cleanly after {mins:.0f} min "
                  f"at {now:,} steps")
            return 0

        # SIGINT reaches the child directly when it shares our terminal, so a
        # child killed by SIGINT means the user stopped it, not a crash.
        if rc in (130, -signal.SIGINT, 128 + signal.SIGINT):
            print(f"\n[{stamp()}] child stopped by Ctrl-C — not restarting")
            return 130

        why = (f"signal {-rc} ({signal.Signals(-rc).name})" if rc < 0
               else f"exit code {rc}")
        progressed = now > seen
        consecutive = 0 if progressed else consecutive + 1
        seen = max(seen, now)

        print(f"\n[{stamp()}] died after {mins:.0f} min with {why}")
        print(f"           progress {now:,}/{target:,} steps"
              f"{'  (+advanced)' if progressed else '  (NO progress)'}")

        if now >= target:
            print(f"[{stamp()}] budget already reached — stopping")
            return 0

        if consecutive > a.max_restarts:
            print(f"[{stamp()}] {consecutive} restarts without progress — "
                  f"giving up.")
            print("           Something is failing before the first "
                  "checkpoint. Run train_3dof.py")
            print("           directly to see the error without the "
                  "supervisor in the way.")
            return 1

        wait = min(a.backoff * (2 ** max(0, consecutive - 1)), 300)
        print(f"[{stamp()}] restarting in {wait:.0f}s "
              f"(consecutive failures: {consecutive}/{a.max_restarts})")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            print(f"\n[{stamp()}] interrupted during backoff — stopping")
            return 130


if __name__ == "__main__":
    sys.exit(main())