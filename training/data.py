"""Data loader: training export -> windows and per frame targets.

Reads the export format and applies the preprocessing in docs/data_format.md.
Needs only numpy. Run from the repo root:

    python training/data.py data/fake/export
    python training/data.py data/export --split user --test-users jamil
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from rule_counter import count_run  # noqa: E402

WINDOW = 60          # frames, same as model.py
STRIDE = 10
TARGET_SIGMA = 2.0   # frames, width of the bump around each rep end
CONDITIONS = ("lighting", "distance_m", "clothing", "speed")


@dataclass
class Run:
    run_id: int
    user: str
    label_count: int
    x: np.ndarray           # (T, 12) float32, preprocessed
    y: np.ndarray           # (T,) float32, rep end target per frame
    t_ms: np.ndarray        # (T,) int64
    rep_end_ms: np.ndarray  # from the rule based counter
    condition: dict = field(default_factory=dict)


def preprocess(lm: np.ndarray) -> np.ndarray:
    """(T, 6, 2) raw landmarks -> (T, 12), centered on the shoulders and scaled by their width.

    The phone must do exactly the same, see the model contract.
    """
    lm = np.asarray(lm, dtype=np.float32)
    mid = (lm[:, 0] + lm[:, 1]) / 2
    width = np.maximum(np.abs(lm[:, 1, 0] - lm[:, 0, 0]), 1e-3)
    return ((lm - mid[:, None, :]) / width[:, None, None]).reshape(len(lm), 12)


def make_target(t_ms: np.ndarray, rep_end_ms, sigma: float = TARGET_SIGMA) -> np.ndarray:
    """1.0 on each rep end frame, falling off as a Gaussian over nearby frames."""
    y = np.zeros(len(t_ms), dtype=np.float32)
    frames = np.arange(len(t_ms))
    for end in rep_end_ms:
        center = int(np.argmin(np.abs(t_ms - end)))
        bump = np.exp(-0.5 * ((frames - center) / sigma) ** 2).astype(np.float32)
        y = np.maximum(y, bump)
    return y


def load_runs(export_dir: Path, tolerance: int = 0) -> tuple[list[Run], list[tuple[int, str]]]:
    """Load labeled runs. Returns (kept runs, [(run_id, reason skipped)]).

    A run is kept when the rule based counter's total is within `tolerance` of
    the hand count, because its rep times become the training labels.
    """
    export_dir = Path(export_dir)
    kept, skipped = [], []
    with open(export_dir / "labels.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            run_id = int(row["run_id"])
            if not row["label_rep_count"]:
                skipped.append((run_id, "no hand count"))
                continue
            with np.load(export_dir / "runs" / f"run_{run_id:05d}.npz") as arrays:
                lm, t_ms = arrays["lm"], arrays["t_ms"]
            if len(lm) < WINDOW:
                skipped.append((run_id, f"only {len(lm)} frames"))
                continue
            counter = count_run([{"t_ms": int(t), "lm": p} for t, p in zip(t_ms, lm.tolist())])
            label = int(row["label_rep_count"])
            if abs(counter.count - label) > tolerance:
                skipped.append((run_id, f"rule count {counter.count} vs hand count {label}"))
                continue
            ends = np.array(counter.rep_end_ms, dtype=np.int64)
            kept.append(Run(run_id, row["user_name"].strip().lower(), label,
                            preprocess(lm), make_target(t_ms, ends), t_ms, ends,
                            {k: row.get(k, "") for k in CONDITIONS}))
    return kept, skipped


def split_by_run(runs: list[Run], val: float = 0.15, test: float = 0.15, seed: int = 67):
    """Random (train, val, test) split where every run lands in exactly one part."""
    if len(runs) < 3:
        raise ValueError(f"need at least 3 runs to split, got {len(runs)}")
    order = np.random.default_rng(seed).permutation(len(runs))
    n_val = max(1, round(len(runs) * val))
    n_test = max(1, round(len(runs) * test))

    def pick(idx):
        return [runs[i] for i in idx]

    return pick(order[n_val + n_test:]), pick(order[:n_val]), pick(order[n_val:n_val + n_test])


def split_by_user(runs: list[Run], test_users: list[str], val: float = 0.15, seed: int = 67):
    """Hold out whole people for testing. Val is a random share of the other runs."""
    held = {u.strip().lower() for u in test_users}
    test = [r for r in runs if r.user in held]
    rest = [r for r in runs if r.user not in held]
    if not test or len(rest) < 2:
        users = sorted({r.user for r in runs})
        raise ValueError(f"cannot hold out {sorted(held)}, users with usable runs: {users}")
    order = np.random.default_rng(seed).permutation(len(rest))
    n_val = max(1, round(len(rest) * val))
    return [rest[i] for i in order[n_val:]], [rest[i] for i in order[:n_val]], test


def make_windows(runs: list[Run], stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Cut runs into (N, 60, 12) inputs and (N, 60) targets."""
    xs, ys = [], []
    for run in runs:
        last = len(run.x) - WINDOW
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)  # do not drop the end of the run
        for s in starts:
            xs.append(run.x[s:s + WINDOW])
            ys.append(run.y[s:s + WINDOW])
    if not xs:
        return np.empty((0, WINDOW, 12), np.float32), np.empty((0, WINDOW), np.float32)
    return np.stack(xs), np.stack(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--split", choices=["run", "user"], default="run")
    ap.add_argument("--test-users", nargs="+", default=[], help="needed with --split user")
    ap.add_argument("--tolerance", type=int, default=0, help="allowed rule count difference")
    args = ap.parse_args()

    runs, skipped = load_runs(args.export_dir, args.tolerance)
    print(f"kept {len(runs)} runs, skipped {len(skipped)}")
    for run_id, reason in skipped:
        print(f"  run {run_id}: {reason}")
    if args.split == "user":
        parts = split_by_user(runs, args.test_users)
    else:
        parts = split_by_run(runs)
    for name, part in zip(("train", "val", "test"), parts):
        x, y = make_windows(part)
        print(f"{name:5} {len(part):3} runs  x {x.shape}  y {y.shape}")


if __name__ == "__main__":
    main()
