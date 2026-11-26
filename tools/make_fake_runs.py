"""Generate synthetic six-seven runs so every member can start before real data exists.

The runs follow docs/data_format.md exactly. The motion is simple (wrists move
up and down in opposite phase at a varying speed, with noise that depends on
the recording condition), so it is good for plumbing and smoke tests, not for
reporting model accuracy. Never mix fake runs into the real dataset.

Examples (run from the repo root)

    # 1. Write payload JSON files, one per run (Member B: see what to send)
    python tools/make_fake_runs.py --n 5 --out data/fake/payloads

    # 2. Upload runs to a running server (Members B and C: fill the leaderboard)
    python tools/make_fake_runs.py --n 20 --post http://localhost:8000

    # 3. Write the training export format directly (Member D: build the loader)
    python tools/make_fake_runs.py --n 60 --export data/fake/export

Only numpy and the standard library are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rule_counter import count_run  # noqa: E402

USERS = ["angie", "april", "jamil", "sean", "friend1", "friend2"]
LIGHTING = ["bright", "dim"]
DISTANCES = [1.5, 2.0, 3.0]
CLOTHING = ["dark", "light", "loose"]
SPEEDS = {"slow": 1.2, "normal": 2.0, "fast": 2.8}  # reps per second


def make_run(rng: np.random.Generator, fps: float = 30.0, duration_s: float = 30.0) -> dict:
    cond = {
        "lighting": str(rng.choice(LIGHTING)),
        "distance_m": float(rng.choice(DISTANCES)),
        "clothing": str(rng.choice(CLOTHING)),
        "speed": str(rng.choice(list(SPEEDS))),
    }
    scale = 2.0 / cond["distance_m"]  # body looks smaller when farther away
    noise = 0.004 * (2.0 if cond["lighting"] == "dim" else 1.0)
    elbow_noise = noise * (2.0 if cond["clothing"] == "loose" else 1.0)

    base_rate = SPEEDS[cond["speed"]] * rng.uniform(0.85, 1.15)
    idle_s = rng.uniform(0.3, 1.0)  # person settles before starting
    cx, cy = 0.5 + rng.uniform(-0.05, 0.05), 0.45 + rng.uniform(-0.05, 0.05)
    half_sw = 0.08 * scale  # half shoulder width
    amp = 0.07 * scale * rng.uniform(0.8, 1.2)  # wrist vertical amplitude

    frames, rep_end_ms = [], []
    phase, t = 0.0, 0.0
    dt_nominal = 1.0 / fps
    while t < duration_s:
        t_ms = int(round(t * 1000))
        if t >= idle_s:
            # speed drifts slowly and tires a bit near the end
            rate = base_rate * (1 + 0.1 * math.sin(t * 0.7)) * (1 - 0.15 * t / duration_s)
            prev = phase
            phase += 2 * math.pi * rate * dt_nominal
            if int(prev // (2 * math.pi)) != int(phase // (2 * math.pi)):
                rep_end_ms.append(t_ms)
        wave = math.sin(phase)
        sway = 0.01 * scale * math.sin(t * 1.3)

        ls = (cx - half_sw + sway, cy)
        rs = (cx + half_sw + sway, cy)
        wrist_y = cy + 0.12 * scale
        lw = (ls[0] - 0.02 * scale, wrist_y - amp * wave)
        rw = (rs[0] + 0.02 * scale, wrist_y + amp * wave)
        le = ((ls[0] + lw[0]) / 2 - 0.04 * scale, (ls[1] + lw[1]) / 2 + 0.03 * scale)
        re = ((rs[0] + rw[0]) / 2 + 0.04 * scale, (rs[1] + rw[1]) / 2 + 0.03 * scale)

        pts = np.array([ls, rs, le, re, lw, rw])
        jitter = rng.normal(0, noise, pts.shape)
        jitter[2:4] = rng.normal(0, elbow_noise, (2, 2))
        pts = np.clip(pts + jitter, 0.0, 1.0)
        frames.append({"t_ms": t_ms, "lm": [[round(float(x), 4), round(float(y), 4)] for x, y in pts]})

        t += dt_nominal * rng.uniform(0.9, 1.1)  # real cameras do not tick evenly

    true_count = len(rep_end_ms)
    return {
        "schema_version": 1,
        "user_name": str(rng.choice(USERS)),
        "device": "synthetic",
        "fps": fps,
        "duration_s": duration_s,
        "app_rep_count": count_run(frames).count,
        "label_rep_count": true_count,
        "condition": cond,
        "frames": frames,
        "_rep_end_ms": rep_end_ms,  # ground truth, only exists in fake data
    }


def payload_only(run: dict) -> dict:
    return {k: v for k, v in run.items() if not k.startswith("_")}


def write_export(runs: list[dict], out: Path) -> None:
    """Write the C -> D training export format (docs/data_format.md, section 'Training export')."""
    (out / "runs").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "user_name", "device", "fps", "duration_s", "n_frames",
                    "app_rep_count", "label_rep_count", "lighting", "distance_m",
                    "clothing", "speed", "created_at"])
        for i, run in enumerate(runs, start=1):
            lm = np.array([f["lm"] for f in run["frames"]], dtype=np.float32)
            t_ms = np.array([f["t_ms"] for f in run["frames"]], dtype=np.int64)
            np.savez_compressed(out / "runs" / f"run_{i:05d}.npz", lm=lm, t_ms=t_ms,
                                rep_end_ms=np.array(run["_rep_end_ms"], dtype=np.int64))
            c = run["condition"]
            w.writerow([i, run["user_name"], run["device"], run["fps"], run["duration_s"],
                        len(lm), run["app_rep_count"], run["label_rep_count"], c["lighting"],
                        c["distance_m"], c["clothing"], c["speed"], "synthetic"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="number of runs")
    ap.add_argument("--seed", type=int, default=67)
    ap.add_argument("--out", type=Path, help="write one payload JSON per run here")
    ap.add_argument("--post", metavar="URL", help="POST runs to this server, e.g. http://localhost:8000")
    ap.add_argument("--export", type=Path, help="write the training export format here")
    args = ap.parse_args()
    if not (args.out or args.post or args.export):
        ap.error("choose at least one of --out, --post, --export")

    rng = np.random.default_rng(args.seed)
    runs = [make_run(rng) for _ in range(args.n)]
    errors = [r["app_rep_count"] - r["label_rep_count"] for r in runs]
    print(f"generated {len(runs)} runs, rule counter error mean {np.mean(errors):+.2f}, "
          f"max abs {max(map(abs, errors))}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for i, run in enumerate(runs, start=1):
            (args.out / f"run_{i:05d}.json").write_text(json.dumps(payload_only(run)), encoding="utf-8")
        print(f"wrote payloads to {args.out}")

    if args.post:
        url = args.post.rstrip("/") + "/runs"
        for run in runs:
            req = urllib.request.Request(url, data=json.dumps(payload_only(run)).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(resp.status, resp.read().decode())

    if args.export:
        write_export(runs, args.export)
        print(f"wrote training export to {args.export}")


if __name__ == "__main__":
    main()
