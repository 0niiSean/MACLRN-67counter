"""Train RepNet, score it on held out runs, and log everything to MLflow.

From the repo root:

    python training/train.py data/export --name v1
    python training/train.py data/export --name v1-jamil --split user --test-users jamil
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse the results

Train and validation use only runs where the rule based counter agrees with the
hand count (its rep times are the labels). The test set keeps every labeled run,
so the comparison with the rule based baseline is fair.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn

import data
import evaluate
from model import RepNet

ROOT = Path(__file__).resolve().parent.parent


def predictor(model: RepNet):
    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(torch.from_numpy(x)).numpy()
    return predict


def fit(model, train_xy, val_xy, epochs: int, batch_size: int, lr: float, patience: int):
    """Train with early stopping on validation loss. Returns the best epoch's loss."""
    x, y = (torch.from_numpy(a) for a in train_xy)
    vx, vy = (torch.from_numpy(a) for a in val_xy)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss, best_state, stale = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(x)).split(batch_size):
            loss = loss_fn(model.logits(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model.logits(vx), vy).item()
        mlflow.log_metrics({"train_loss": total / len(x), "val_loss": val_loss}, step=epoch)
        print(f"epoch {epoch:3}  train {total / len(x):.4f}  val {val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise SystemExit("training produced no finite validation loss")
    model.load_state_dict(best_state)
    return best_loss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("--name", required=True, help="run name, also names the weights file")
    ap.add_argument("--split", choices=["run", "user"], default="run")
    ap.add_argument("--test-users", nargs="+", default=[])
    ap.add_argument("--tolerance", type=int, default=0, help="allowed rule count difference for training")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=32,
                    help="export_tflite.py expects the default of 32")
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    runs, skipped = data.load_runs(args.export_dir, tolerance=sys.maxsize)
    if args.split == "user":
        train, val, test = data.split_by_user(runs, args.test_users, seed=args.seed)
    else:
        train, val, test = data.split_by_run(runs, seed=args.seed)

    def agrees(r):
        return abs(len(r.rep_end_ms) - r.label_count) <= args.tolerance

    train, val = [r for r in train if agrees(r)], [r for r in val if agrees(r)]
    if not train or not val:
        sys.exit("no train or validation runs left, collect more runs or raise --tolerance")
    print(f"runs: train {len(train)}, val {len(val)}, test {len(test)}, unusable {len(skipped)}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI",
                                           f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"))
    mlflow.set_experiment("67counter")
    with mlflow.start_run(run_name=args.name) as active:
        mlflow.log_params({
            **{k: str(v) for k, v in vars(args).items()},
            "window": data.WINDOW, "stride": data.STRIDE, "target_sigma": data.TARGET_SIGMA,
            "read_index": evaluate.READ_INDEX, "threshold": evaluate.THRESHOLD,
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
        })
        mlflow.set_tag("test_run_ids", " ".join(str(r.run_id) for r in test))

        model = RepNet(args.channels)
        best = fit(model, data.make_windows(train), data.make_windows(val),
                   args.epochs, args.batch_size, args.lr, args.patience)
        metrics, rows = evaluate.evaluate(predictor(model), test)
        mlflow.log_metrics({"best_val_loss": best, **{f"test_{k}": v for k, v in metrics.items()}})

        out = ROOT / "models"
        out.mkdir(exist_ok=True)
        weights = out / f"repnet_{args.name}.pt"
        torch.save(model.state_dict(), weights)
        results = out / f"test_{args.name}.csv"
        with open(results, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        mlflow.log_artifact(str(weights))
        mlflow.log_artifact(str(results))

    print(f"\ntest set ({len(test)} runs)       model    rule baseline")
    for key in ("count_mae", "count_bias", "within_one"):
        print(f"  {key:22} {metrics['model_' + key]:7.3f}  {metrics['rule_' + key]:7.3f}")
    for key in ("lighting", "distance_m", "clothing", "speed"):
        for value, m in evaluate.by_condition(rows, key).items():
            print(f"  {key}={value:8} runs {m['runs']:3}  model MAE {m['count_mae']:.2f}")
    print(f"\nweights {weights.relative_to(ROOT)}   MLflow run id {active.info.run_id}")


if __name__ == "__main__":
    main()
