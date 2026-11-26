"""Turn model outputs into rep counts and score them against hand counts.

Needs only numpy (plus ai-edge-litert to score a .tflite file). From the repo root:

    python training/evaluate.py models/67counter_v1.tflite data/export

Note that this CLI scores every labeled run in the folder, including training
runs. The held out test numbers come from train.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from data import WINDOW, Run, load_runs

# Decoding, see the model contract in docs/data_format.md
READ_INDEX = 54
THRESHOLD = 0.5
MIN_GAP_MS = 150


def windows_for_stream(x: np.ndarray) -> np.ndarray:
    """(T, 12) -> (T, 60, 12): window i puts frame i at READ_INDEX.

    The start is padded by repeating the first frame and the end by repeating the
    last one, so reps in the first and last moments of a run can still be counted.
    """
    padded = np.concatenate([
        np.repeat(x[:1], READ_INDEX, axis=0),
        x,
        np.repeat(x[-1:], WINDOW - 1 - READ_INDEX, axis=0),
    ])
    return np.stack([padded[i:i + WINDOW] for i in range(len(x))])


def decode(probs: np.ndarray, t_ms: np.ndarray, threshold: float = THRESHOLD,
           min_gap_ms: int = MIN_GAP_MS) -> list[int]:
    """Rep times: frames where the probability rises across the threshold."""
    reps: list[int] = []
    previous = 0.0
    for p, t in zip(probs, t_ms):
        if p >= threshold > previous and (not reps or t - reps[-1] >= min_gap_ms):
            reps.append(int(t))
        previous = p
    return reps


def count_with_model(predict, run: Run) -> int:
    """`predict` maps (N, 60, 12) float32 to (N, 60) probabilities."""
    probs = predict(windows_for_stream(run.x))[:, READ_INDEX]
    return len(decode(probs, run.t_ms))


def score(counts, labels) -> dict[str, float]:
    error = np.asarray(counts, dtype=float) - np.asarray(labels, dtype=float)
    return {
        "count_mae": float(np.abs(error).mean()),
        "count_bias": float(error.mean()),
        "within_one": float((np.abs(error) <= 1).mean()),
    }


def evaluate(predict, runs: list[Run]) -> tuple[dict[str, float], list[dict]]:
    """Score the model and the rule based baseline on the same runs."""
    rows = [{
        "run_id": r.run_id, "user": r.user, **r.condition,
        "hand_count": r.label_count,
        "model_count": count_with_model(predict, r),
        "rule_count": len(r.rep_end_ms),
    } for r in runs]
    labels = [row["hand_count"] for row in rows]
    metrics = {}
    for name in ("model", "rule"):
        for key, value in score([row[f"{name}_count"] for row in rows], labels).items():
            metrics[f"{name}_{key}"] = value
    return metrics, rows


def by_condition(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    out = {}
    for value in sorted({row.get(key, "") for row in rows} - {""}):
        group = [row for row in rows if row.get(key) == value]
        out[value] = {"runs": len(group), **score([r["model_count"] for r in group],
                                                  [r["hand_count"] for r in group])}
    return out


def tflite_predictor(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]["index"]
    out = interpreter.get_output_details()[0]["index"]

    def predict(x: np.ndarray) -> np.ndarray:
        results = []
        for window in x:
            interpreter.set_tensor(inp, window[None])
            interpreter.invoke()
            results.append(interpreter.get_tensor(out)[0])
        return np.stack(results)

    return predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help=".tflite file")
    ap.add_argument("export_dir", type=Path)
    args = ap.parse_args()

    runs, skipped = load_runs(args.export_dir, tolerance=sys.maxsize)
    if not runs:
        sys.exit(f"no usable labeled runs ({len(skipped)} skipped)")
    metrics, _ = evaluate(tflite_predictor(args.model), runs)
    print(f"{len(runs)} runs")
    for key, value in metrics.items():
        print(f"  {key:18} {value:.3f}")


if __name__ == "__main__":
    main()
