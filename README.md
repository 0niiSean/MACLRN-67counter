# 67counter

A two tier machine learning system that measures how fast you do the "six-seven" motion.
Built for MAC-LRN (Machine Learning), End-of-Term Project, 1T2627.

## Problem

People do the six-seven motion as fast as they can, one hand up while the other goes down, then switching.
67counter lets you stand in front of a phone, do the motion for 30 seconds, and get your speed in reps per minute.
Scores go on a leaderboard so friends can compete.

## Architecture

```
Tier 1 (Mobile, Android)                 Tier 2 (Edge server, laptop)
------------------------                 ----------------------------
Camera -> MediaPipe Pose                 FastAPI  -> SQLite (runs, users)
  -> landmarks (shoulders,   -- JSON ->  Leaderboard endpoint
     elbows, wrists)                     PyTorch training + MLflow tracking
  -> rep counter (+ TFLite model) <-     TFLite export -> /model/latest
  -> speed shown on screen
```

Only landmark coordinates leave the phone. No video is uploaded.

## Repository layout

| Folder | Purpose |
| --- | --- |
| `app/` | Android app (Kotlin, MediaPipe Pose, TFLite) |
| `server/` | FastAPI service and SQLite storage |
| `training/` | PyTorch training, MLflow tracking, TFLite export |
| `tools/` | Shared rule based counter and synthetic run generator |
| `docs/` | Data format and contracts, contributing guide, report notes |
| `data/` | Local datasets (not committed) |
| `models/` | Exported models (not committed) |

## Start here

1. Read `docs/CONTRIBUTING.md` for roles, setup, and the "start here" table for your role.
2. Read `docs/data_format.md`. It holds the three contracts between the app, server, and training code.
3. No real data yet? `tools/make_fake_runs.py` produces synthetic runs in the exact formats.

## Quick start (server)

```bash
cd server
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive API.

## Team

Angie Olonan, April Denise Besa, Jamil So, Sean Andrei Olores

See `docs/CONTRIBUTING.md` for roles and commit conventions. GitHub Actions runs the server and contract tests on every pull request.
