# Server (Tier 2)

FastAPI service on the laptop acting as the edge server.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check |
| `POST /runs` | Receive a run (landmarks, counts, conditions) |
| `PATCH /runs/{id}/label` | Add the hand counted label later |
| `GET /leaderboard` | Best run per user |
| `GET /model/latest` | Download the newest TFLite model |

## Run it

```bash
cd server
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pytest
```

Runs are stored in `server/runs.db` (not committed). Delete it to start fresh.

To fill the leaderboard with synthetic runs while the app is being built, run this from the repo root:

```bash
pip install numpy
python tools/make_fake_runs.py --n 20 --post http://localhost:8000
```

Use a separate copy of `runs.db` for fake data, and never export fake runs as training data.

## Reaching the laptop from the phone

1. Put both on the same Wi-Fi (phone hotspots work if campus Wi-Fi blocks device to device traffic).
2. Find the laptop IP with `ipconfig` (Windows) or `ip a` (Linux).
3. On Windows, set the Wi-Fi network profile to Private and allow port 8000 in an administrator terminal:

   ```
   netsh advfirewall firewall add rule name="67counter 8000" dir=in action=allow protocol=TCP localport=8000
   ```

4. Open `http://<laptop-ip>:8000/docs` in the phone browser. If it loads, the app can reach the server.

## To do (Member C)

- [ ] `export_runs.py`: write labeled runs in the training export format from `docs/data_format.md`, and share a sample with D
- [ ] Any extra tables or fields the team needs (for example users), with tests
- [ ] MLflow tracking server (`mlflow ui`) and the model naming rule agreed with D
- [ ] Copy each approved `.tflite` into `models/`
- [ ] Demo network tested on the demo laptop and phone
- [ ] Server notes for the report
