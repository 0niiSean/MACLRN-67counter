"""Run from the repo root:  python -m pytest training/test_evaluate.py"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate  # noqa: E402


def test_stream_windows_put_each_frame_at_read_index():
    x = np.arange(100 * 12, dtype=np.float32).reshape(100, 12)
    w = evaluate.windows_for_stream(x)
    assert w.shape == (100, 60, 12)
    np.testing.assert_array_equal(w[:, evaluate.READ_INDEX], x)
    np.testing.assert_array_equal(w[0, 0], x[0])    # start padding
    np.testing.assert_array_equal(w[-1, -1], x[-1])  # end padding


def test_decode_counts_rising_edges_with_min_gap():
    t = np.arange(10) * 33
    probs = np.array([0.1, 0.9, 0.9, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.6])
    # frame 4 is only 99 ms after the rep at frame 1, so it is ignored
    assert evaluate.decode(probs, t) == [33, 297]


def test_score():
    m = evaluate.score([10, 12, 20], [10, 11, 17])
    assert np.isclose(m["count_mae"], 4 / 3) and np.isclose(m["within_one"], 2 / 3)
