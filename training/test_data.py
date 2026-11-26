"""Run from the repo root:  python -m pytest training/test_data.py"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

import data  # noqa: E402
from make_fake_runs import make_run, write_export  # noqa: E402


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    rng = np.random.default_rng(3)
    write_export([make_run(rng) for _ in range(20)], out)
    return out


def test_preprocess_matches_contract():
    lm = np.array([[[0.4, 0.3], [0.6, 0.3], [0.4, 0.4], [0.6, 0.4], [0.4, 0.5], [0.6, 0.7]]])
    x = data.preprocess(lm)
    assert x.shape == (1, 12) and x.dtype == np.float32
    np.testing.assert_allclose(x[0, :4], [-0.5, 0, 0.5, 0], atol=1e-6)
    np.testing.assert_allclose(x[0, 10:], [0.5, 2.0], atol=1e-6)


def test_target_peaks_at_rep_ends():
    t = np.arange(100) * 33
    y = data.make_target(t, [330, 1650])
    assert y[10] == y[50] == 1.0 and y[30] < 0.01


def test_load_keeps_only_matching_runs(export):
    runs, skipped = data.load_runs(export)
    assert runs and len(runs) + len(skipped) == 20
    for r in runs:
        assert len(r.rep_end_ms) == r.label_count
        assert len(r.x) == len(r.y) == len(r.t_ms)


def test_unlabeled_run_is_skipped(tmp_path):
    write_export([make_run(np.random.default_rng(1))], tmp_path)
    path = tmp_path / "labels.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows[0]["label_rep_count"] = ""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    runs, skipped = data.load_runs(tmp_path)
    assert not runs and skipped == [(1, "no hand count")]


def test_splits_do_not_leak(export):
    runs, _ = data.load_runs(export, tolerance=2)
    parts = data.split_by_run(runs)
    ids = [r.run_id for part in parts for r in part]
    assert sorted(ids) == sorted(r.run_id for r in runs) and all(parts)

    user = runs[0].user
    train, val, test = data.split_by_user(runs, [user])
    assert all(r.user == user for r in test)
    assert all(r.user != user for r in train + val)
    with pytest.raises(ValueError):
        data.split_by_user(runs, ["nobody"])


def test_windows_cover_run_end(export):
    runs, _ = data.load_runs(export)
    x, y = data.make_windows(runs[:1])
    assert x.shape[1:] == (60, 12) and y.shape == x.shape[:2]
    np.testing.assert_array_equal(x[-1], runs[0].x[-60:])
