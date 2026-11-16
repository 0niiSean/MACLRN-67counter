import os
import tempfile
from pathlib import Path

import main
from fastapi.testclient import TestClient


def make_run(user, reps):
    frame = {"t_ms": 0, "lm": [[0.5, 0.5]] * 6}
    return {
        "user_name": user, "fps": 30, "duration_s": 30.0,
        "app_rep_count": reps, "frames": [frame],
    }


def test_upload_and_leaderboard():
    main.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
    with TestClient(main.app) as client:
        assert client.post("/runs", json=make_run("a", 50)).status_code == 201
        assert client.post("/runs", json=make_run("b", 60)).status_code == 201
        board = client.get("/leaderboard").json()
        assert board[0]["user_name"] == "b"
        assert board[0]["best_rpm"] == 120.0


def test_bad_landmarks_rejected():
    main.DB_PATH = Path(tempfile.mkdtemp()) / "test.db"
    bad = make_run("a", 10)
    bad["frames"] = [{"t_ms": 0, "lm": [[0.5, 0.5]]}]
    with TestClient(main.app) as client:
        assert client.post("/runs", json=bad).status_code == 422


def test_model_404_when_missing():
    main.MODEL_DIR = Path(tempfile.mkdtemp())
    with TestClient(main.app) as client:
        assert client.get("/model/latest").status_code == 404
