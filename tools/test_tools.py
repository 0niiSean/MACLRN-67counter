"""Checks that the shared tools still match the data contract.

Run from the repo root:  python -m pytest tools
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "server"))

import main  # noqa: E402  (server/main.py)
from make_fake_runs import make_run, payload_only, write_export  # noqa: E402
from rule_counter import count_run  # noqa: E402


def test_rule_counter_close_to_truth_on_fake_runs():
    rng = np.random.default_rng(0)
    for _ in range(20):
        run = make_run(rng)
        assert abs(count_run(run["frames"]).count - run["label_rep_count"]) <= 2


def _count(signs):
    """Count a motion given as +1 (left hand up) and -1 (right hand up), 10 frames each."""
    frames = []
    for i, sign in enumerate(s for s in signs for _ in range(10)):
        lm = [[0.4, 0.4], [0.6, 0.4], [0.35, 0.5], [0.65, 0.5],
              [0.35, 0.5 - 0.1 * sign], [0.65, 0.5 + 0.1 * sign]]
        frames.append({"t_ms": i * 33, "lm": lm})
    return count_run(frames).count


def test_every_left_hand_top_counts_once():
    assert _count([1, -1, 1, -1]) == 2      # starts left: the first top counts
    assert _count([-1, 1, -1, 1]) == 2      # starts right
    assert _count([1, -1, 1]) == 2          # ends on a top
    assert _count([0, 0]) == 0              # standing still
    assert _count([1, 0, 1, 0, 1]) == 1     # right hand never goes up, no recount


def test_fake_payload_accepted_by_server():
    main.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
    run = payload_only(make_run(np.random.default_rng(1)))
    with TestClient(main.app) as client:
        resp = client.post("/runs", json=run)
        assert resp.status_code == 201, resp.text


def test_export_layout():
    out = Path(tempfile.mkdtemp())
    runs = [make_run(np.random.default_rng(i)) for i in range(3)]
    write_export(runs, out)
    assert (out / "labels.csv").exists()
    data = np.load(out / "runs" / "run_00001.npz")
    assert data["lm"].shape[1:] == (6, 2) and data["lm"].dtype == np.float32
    assert data["t_ms"].shape[0] == data["lm"].shape[0]
