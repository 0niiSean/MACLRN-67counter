# Training

The model marks the moment each rep ends. Its input, output, preprocessing, and file naming are fixed in the model contract in `docs/data_format.md`.

## Setup

Training, evaluation, and checking `.tflite` files run natively, including on Windows. Use Python 3.11 to 3.13. From the repo root:

```bash
cd training
python -m venv .venv              # Windows with several versions: py -3.13 -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## TFLite export

The converter (`litert-torch`, formerly `ai-edge-torch`) has no Windows build, so the export runs in Docker. Start Docker Desktop, then from the repo root in PowerShell (the image takes about 3 GB):

```powershell
docker build -t 67counter-export training
docker run --rm -v "${PWD}:/repo" 67counter-export python export_tflite.py --out ../models/67counter_v0.tflite
training\.venv\Scripts\python training\check_tflite.py models\67counter_v0.tflite
```

Add `--weights <state_dict file>` (a path inside the repo, relative to `training/`) to export a trained model. The export fails if the TFLite output differs from PyTorch by more than 1e-4.

| File | Purpose |
| --- | --- |
| `data.py` | Loads the export, preprocesses, labels rep ends, cuts windows, splits by run or person |
| `train.py` | Trains, scores on held out runs, logs to MLflow, saves weights to `models/` |
| `evaluate.py` | Model output to rep count, metrics, and scoring of a `.tflite` file |
| `model.py` | `RepNet`, a small 1D CNN with the contract input and output shapes |
| `export_tflite.py` | PyTorch to `.tflite`, with a parity check |
| `check_tflite.py` | Checks shapes and output range, lists the ops, and times inference |

## Train, export, check

From the repo root, with the training venv active:

```bash
python training/train.py data/export --name v1
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then export the weights (PowerShell, Docker running) and score the exported file:

```powershell
docker run --rm -v "${PWD}:/repo" 67counter-export python export_tflite.py --weights ../models/repnet_v1.pt --out ../models/67counter_v1.tflite
python training/check_tflite.py models/67counter_v1.tflite
python training/evaluate.py models/67counter_v1.tflite data/export
```

Report the test set numbers printed by `train.py`. `evaluate.py` scores every labeled run, including training runs.

## Pipeline

1. **Prove the export first.** Done with the untrained dummy model `67counter_v0.tflite` (laptop check only, still to be tested on the phone).
2. **Data loader.** `data.py` reads the training export (`labels.csv` plus `runs/*.npz`): `python training/data.py data/fake/export`. Until real runs exist, use synthetic data: `python tools/make_fake_runs.py --n 60 --export data/fake/export` from the repo root.
3. **Labels.** Run `tools/rule_counter.py` on each run to get `rep_end_ms`, and keep only runs where its total matches `label_rep_count`. Turn each rep end into a positive frame (or a small Gaussian bump) in a per frame target.
4. **Windows and splits.** Cut 60 frame windows with a stride, and split train and test by person or by run so the test set is never seen in training.
5. **Model.** A small 1D CNN or GRU with a sigmoid per frame.
6. **Tracking.** Log parameters, metrics, and the model file to MLflow for every run.
7. **Export.** Name it `67counter_v<N>.tflite` and give it to Member C for `models/`, then test on the phone with Member A.

## Evaluation for the report

| Metric | How |
| --- | --- |
| Count error | Mean absolute error of model count against `label_rep_count`, next to the rule based baseline |
| Within one rep | Share of test runs where the count is off by at most one |
| Per condition | Count error broken down by lighting, distance, clothing, and speed |
| Latency | Inference time per window on the demo phone (measured with Member A) |

## Released models

| Version | MLflow run id | Count MAE (test) | Notes |
| --- | --- | --- | --- |
| | | | |
