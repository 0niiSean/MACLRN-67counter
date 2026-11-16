# Training

Planned pipeline

1. Read labeled runs from `server/runs.db`.
2. Window the landmark sequences and train a small PyTorch model.
3. Log parameters, metrics, and the model to MLflow (`mlflow ui`).
4. Export to TFLite and copy the file into `models/`.
5. The server serves it from `GET /model/latest`.

Open decision, what the model predicts (corrected rep count or valid rep classification). Record the choice here once the team agrees.

Test the PyTorch to TFLite export early with a dummy model, since this step is the most likely to break.
