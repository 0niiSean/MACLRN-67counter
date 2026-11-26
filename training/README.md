# Training

The model marks the moment each rep ends. Its input, output, preprocessing, and file naming are fixed in the model contract in `docs/data_format.md`.

## Setup

PyTorch and the TFLite converters may not have wheels for the newest Python, so use Python 3.11 or 3.12.

```bash
cd training
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`ai-edge-torch` may not install on native Windows. If it fails, use WSL or Google Colab for the export step only.

## Pipeline

1. **Prove the export first.** Export a tiny dummy model with input `[1, 60, 12]` and output `[1, 60]` to `.tflite`, and run it with the TFLite interpreter on the laptop. Try `ai-edge-torch` first, then PyTorch to ONNX to TFLite. If both fail, train the same small model in Keras.
2. **Data loader.** Read the training export (`labels.csv` plus `runs/*.npz`). Until real runs exist, use synthetic data: `python tools/make_fake_runs.py --n 60 --export data/fake/export` from the repo root.
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
