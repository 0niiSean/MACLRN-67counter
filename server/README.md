# Server (Tier 2)

FastAPI service on the laptop acting as the edge server.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check |
| `POST /runs` | Receive a run (landmarks, counts, conditions) |
| `PATCH /runs/{id}/label` | Add the hand counted label later |
| `GET /leaderboard` | Best run per user |
| `GET /model/latest` | Download the newest TFLite model |

Run tests with `pytest` from this folder.

To let the phone reach the laptop, put both on the same Wi-Fi, find the laptop IP (`ipconfig` or `ip a`), and use `http://<laptop-ip>:8000` in the app.
