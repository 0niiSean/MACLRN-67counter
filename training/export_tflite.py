"""Export RepNet to TFLite and check it against PyTorch.

Runs on Linux or macOS only (the converter has no Windows build), see README.md.

    python export_tflite.py --out ../models/67counter_v0.tflite                 # dummy weights
    python export_tflite.py --weights repnet.pt --out ../models/67counter_v1.tflite
"""

import argparse
from pathlib import Path

import litert_torch
import numpy as np
import torch

from model import FEATURES, WINDOW, RepNet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, help="state_dict file, omit for an untrained dummy model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(67)
    model = RepNet()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()

    sample = torch.randn(1, WINDOW, FEATURES)
    edge_model = litert_torch.convert(model, (sample,))

    with torch.no_grad():
        expected = model(sample).numpy()
    got = np.asarray(edge_model(sample))
    max_diff = float(np.abs(expected - got).max())
    if got.shape != (1, WINDOW) or max_diff > 1e-4:
        raise SystemExit(f"export mismatch: shape {got.shape}, max diff {max_diff}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB), max diff vs PyTorch {max_diff:.2e}")


if __name__ == "__main__":
    main()
