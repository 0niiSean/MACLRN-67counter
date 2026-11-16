"""67counter edge server (Tier 2).

Receives landmark runs from the Android app, stores them in SQLite,
serves the leaderboard, and serves the latest TFLite model.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent / "runs.db"
MODEL_DIR = Path(__file__).parent.parent / "models"

app = FastAPI(title="67counter server", version="0.1.0")


# ---------- Schemas ----------

class Frame(BaseModel):
    t_ms: int
    lm: List[List[float]] = Field(..., description="6 [x, y] pairs in the fixed landmark order")


class Condition(BaseModel):
    lighting: Optional[str] = None
    distance_m: Optional[float] = None
    clothing: Optional[str] = None
    speed: Optional[str] = None


class RunIn(BaseModel):
    schema_version: int = 1
    user_name: str
    device: Optional[str] = None
    fps: float
    duration_s: float = Field(..., gt=0)
    app_rep_count: int = Field(..., ge=0)
    label_rep_count: Optional[int] = Field(None, ge=0)
    condition: Optional[Condition] = None
    frames: List[Frame]


# ---------- Database ----------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                user_name TEXT NOT NULL,
                device TEXT,
                fps REAL NOT NULL,
                duration_s REAL NOT NULL,
                app_rep_count INTEGER NOT NULL,
                label_rep_count INTEGER,
                reps_per_min REAL NOT NULL,
                condition_json TEXT,
                frames_json TEXT NOT NULL
            )
            """
        )


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/runs", status_code=201)
def create_run(run: RunIn):
    for f in run.frames:
        if len(f.lm) != 6 or any(len(p) != 2 for p in f.lm):
            raise HTTPException(422, "each frame needs 6 [x, y] landmark pairs")
    rpm = run.app_rep_count / run.duration_s * 60
    with get_db() as db:
        cur = db.execute(
            """
            INSERT INTO runs (created_at, user_name, device, fps, duration_s,
                              app_rep_count, label_rep_count, reps_per_min,
                              condition_json, frames_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                run.user_name,
                run.device,
                run.fps,
                run.duration_s,
                run.app_rep_count,
                run.label_rep_count,
                rpm,
                json.dumps(run.condition.model_dump()) if run.condition else None,
                json.dumps([f.model_dump() for f in run.frames]),
            ),
        )
        run_id = cur.lastrowid
    return {"id": run_id, "reps_per_min": round(rpm, 2)}


@app.get("/leaderboard")
def leaderboard(limit: int = 20):
    """Best run per user, highest reps per minute first."""
    with get_db() as db:
        rows = db.execute(
            """
            SELECT user_name, MAX(reps_per_min) AS best_rpm, COUNT(*) AS runs
            FROM runs
            GROUP BY user_name
            ORDER BY best_rpm DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {"rank": i + 1, "user_name": r["user_name"],
         "best_rpm": round(r["best_rpm"], 2), "runs": r["runs"]}
        for i, r in enumerate(rows)
    ]


@app.patch("/runs/{run_id}/label")
def set_label(run_id: int, label_rep_count: int):
    """Attach a hand counted label to an existing run."""
    with get_db() as db:
        cur = db.execute(
            "UPDATE runs SET label_rep_count = ? WHERE id = ?",
            (label_rep_count, run_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "run not found")
    return {"id": run_id, "label_rep_count": label_rep_count}


@app.get("/model/latest")
def latest_model():
    """Serve the newest exported TFLite model so the phone can download it."""
    files = sorted(MODEL_DIR.glob("*.tflite"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise HTTPException(404, "no model has been exported yet")
    return FileResponse(files[-1], media_type="application/octet-stream", filename=files[-1].name)
