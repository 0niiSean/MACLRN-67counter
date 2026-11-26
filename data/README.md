# Data

Raw data is not committed to Git. Runs are stored in `server/runs.db` and shared through the team drive. Recording videos (used only to settle hand counts) go in the 67 Recordings Drive folder and are kept until the final paper is submitted.

## Layout (local only)

| Path | What |
| --- | --- |
| `data/export/` | Real training export from the server, see `docs/data_format.md` |
| `data/fake/` | Synthetic runs from `tools/make_fake_runs.py`, never used for report numbers |

## Recording checklist

Aim for 100 or more runs in total, about 25 per member, each hand counted by the recorder and a second person. Spread your own 25 runs across these values and fill `condition` in the upload.

| Variable | Values to cover |
| --- | --- |
| Lighting | bright, dim |
| Distance | 1.5 m, 2 m, 3 m |
| Clothing | dark, light, loose |
| Speed | slow, normal, fast |
| People | all four members, plus friends if possible |

## Labeling

Use the rep definition in `docs/data_format.md`. Add the agreed count with `PATCH /runs/{id}/label` (or from the `/docs` page). If the two counts differ by more than one, replay the video before labeling.
