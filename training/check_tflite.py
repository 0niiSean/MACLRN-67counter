"""Check a .tflite file against the model contract and time it. Works on Windows.

    pip install ai-edge-litert numpy
    python training/check_tflite.py models/67counter_v0.tflite
"""

import sys
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python check_tflite.py <model.tflite>")
    interpreter = Interpreter(model_path=sys.argv[1])
    interpreter.allocate_tensors()
    inp, = interpreter.get_input_details()
    out, = interpreter.get_output_details()
    for name, d, shape in (("input", inp, [1, 60, 12]), ("output", out, [1, 60])):
        print(f"{name:6} {d['shape'].tolist()} {d['dtype'].__name__}")
        if d["shape"].tolist() != shape or d["dtype"] != np.float32:
            sys.exit(f"FAIL: {name} should be float32 {shape}")

    x = np.random.default_rng(67).standard_normal((1, 60, 12)).astype(np.float32)
    times = []
    for _ in range(200):
        start = time.perf_counter()
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out["index"])
        times.append((time.perf_counter() - start) * 1000)
    if not (np.isfinite(y).all() and (y >= 0).all() and (y <= 1).all()):
        sys.exit("FAIL: outputs are not probabilities")

    ops = sorted({o["op_name"] for o in interpreter._get_ops_details()} - {"DELEGATE"})
    print("ops   ", ", ".join(ops))
    print(f"OK, median latency on this machine {np.median(times):.3f} ms")


if __name__ == "__main__":
    main()
