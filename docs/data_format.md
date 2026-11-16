# Run data format (v1)

This is the contract between the app, the server, and the training code. Change it only with a team agreement and bump `schema_version`.

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
