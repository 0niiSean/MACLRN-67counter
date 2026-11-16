# Android app (Tier 1)

Planned stack: Kotlin, CameraX, MediaPipe Pose Landmarker, TFLite.

Milestones

1. Camera preview with pose landmarks drawn.
2. Rep counter on wrist and shoulder landmarks, shown live.
3. 30 second timer and result screen with reps per minute.
4. Upload the run as JSON to the server (see `docs/data_format.md`).
5. Download and load the latest TFLite model.

Put the server address in one config constant so it is easy to change at the demo.
