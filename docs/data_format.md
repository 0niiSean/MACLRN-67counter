# Run data format (v1)

This is the contract between the app, the server, and the training code. Change it only with a team agreement and bump `schema_version`.

It defines the three interfaces between the four areas.

| Interface | From | To | Section |
| --- | --- | --- | --- |
| Run upload | B (app) | C (server) | [Upload payload](#upload-payload-post-runs) |
| Training data | C (server) | D (training) | [Training export](#training-export-c-to-d) |
| Model delivery | D (training) | A and B (app) | [Model contract](#model-contract-d-to-a-and-b) |

## Sign-off

Status: **draft, waiting for team sign-off.** Each member ticks their box in the pull request that changes this line to "agreed".

- [ ] A
- [ ] B
- [ ] C
- [ ] D

## Rep definition (proposed)

**One rep is counted each time the left hand reaches the top** (the right hand is down). The first time counts, even if the run starts with the left hand going up. The left hand must come down again, with the right hand up, before it can count again.

When hand counting, count the left hand tops that happen within the 30 seconds. When counts differ by more than one, replay the video.

The rule based counter in `tools/rule_counter.py` implements this definition, and the Kotlin port on the phone must match it.

## Landmarks

Six MediaPipe Pose landmarks are recorded per frame, in this fixed order.

| Index | Name | MediaPipe id |
| --- | --- | --- |
| 0 | left_shoulder | 11 |
| 1 | right_shoulder | 12 |
| 2 | left_elbow | 13 |
| 3 | right_elbow | 14 |
| 4 | left_wrist | 15 |
| 5 | right_wrist | 16 |

Coordinates are MediaPipe normalized values (x and y in 0 to 1, relative to image width and height).

## Upload payload (POST /runs)

```json
{
  "schema_version": 1,
  "user_name": "sean",
  "device": "Pixel 6",
  "fps": 30,
  "duration_s": 30.0,
  "app_rep_count": 52,
  "label_rep_count": 50,
  "condition": {
    "lighting": "bright",
    "distance_m": 2.0,
    "clothing": "dark",
    "speed": "fast"
  },
  "frames": [
    {
      "t_ms": 0,
      "lm": [[0.41, 0.30], [0.59, 0.30], [0.38, 0.45], [0.62, 0.45], [0.36, 0.60], [0.64, 0.58]]
    }
  ]
}
```

Field notes

`app_rep_count` is what the phone counted live. `label_rep_count` is the hand count and is optional at upload, since it may be added later by the person labeling the data. `condition` records the recording variation so we can report results per condition. `lm` holds six `[x, y]` pairs in the order above.

## Derived values

Reps per minute is `app_rep_count / duration_s * 60`. The leaderboard uses the app count for now and will switch to the model corrected count once the model is deployed.

## Training export (C to D)

The server export script writes labeled runs to a folder under `data/` (never committed). By default it exports only runs with a `label_rep_count`.

```
data/export/
  labels.csv
  runs/
    run_00001.npz
    run_00002.npz
    ...
```

`labels.csv` has one row per run, and `run_id` matches the file name.

```
run_id,user_name,device,fps,duration_s,n_frames,app_rep_count,label_rep_count,lighting,distance_m,clothing,speed,created_at
```

Each `run_XXXXX.npz` holds two arrays (load with `numpy.load`).

| Key | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `lm` | (T, 6, 2) | float32 | Landmarks per frame in the order above, raw normalized coordinates |
| `t_ms` | (T,) | int64 | Frame time in ms from the start of recording |

Synthetic exports from `tools/make_fake_runs.py --export` also include `rep_end_ms`, the true time of each rep end. Real exports do not have it, so per rep timestamps come from running the rule based counter.

## Model contract (D to A and B)

The model marks the moment each rep ends. Phone and training code must do the same preprocessing, so any change here needs A and D to agree.

**Preprocessing per frame.** Take the six landmarks, subtract the shoulder midpoint, and divide by the shoulder width.

```
mid = (lm[0] + lm[1]) / 2
sw  = max(|lm[1].x - lm[0].x|, 1e-3)
x'  = (x - mid.x) / sw,   y' = (y - mid.y) / sw
```

Then flatten to 12 values in landmark order: `[ls_x, ls_y, rs_x, rs_y, le_x, le_y, re_x, re_y, lw_x, lw_y, rw_x, rw_y]`.

| | Value |
| --- | --- |
| Input | float32 `[1, 60, 12]`, the last 60 frames (about 2 s at 30 fps), oldest first |
| Output | float32 `[1, 60]`, probability that a rep ends at each frame of the window |
| Frame rate | Trained at 30 fps. If the phone runs much slower, A and D agree on resampling with `t_ms` |
| Suggested decoding | Run once per new frame, read the output for frame index 54 (about 0.2 s of lookahead), count a rep when it crosses 0.5, and skip 150 ms after each rep |

**Naming and delivery.** Files are named `67counter_v<N>.tflite` with `N` going up by one per release, and `N` matches the MLflow run that produced it (record the run id in the MLflow run tags and in `training/README.md`). Member C copies the approved file into `models/`, and `GET /model/latest` serves the newest one. The file name in the response tells the app which version it has.
