# Android app (Tier 1)

Stack: Kotlin, CameraX, MediaPipe Pose Landmarker, TFLite.

Create the project in Android Studio (Empty Views Activity, Kotlin, minimum SDK 24 or higher) with this folder as the project root. Start from the official MediaPipe Pose Landmarker Android example for the camera and landmarker setup, and copy the library versions it uses.

## File split between A and B

A and B work in the same project, so each file has one owner. Suggested package: `com.sixtyseven.counter`.

| Path (under `app/src/main/java/com/sixtyseven/counter/`) | Owner | What it holds |
| --- | --- | --- |
| `pose/PoseTracker.kt` | A | MediaPipe setup, returns a `LandmarkFrame` per camera frame |
| `pose/LandmarkFrame.kt` | A (agreed with B) | The shared frame type below |
| `counter/RepCounter.kt` | A (agreed with B) | The shared counter interface below |
| `counter/RuleCounter.kt` | A | Kotlin port of `tools/rule_counter.py` |
| `counter/ModelCounter.kt` | A | TFLite inference, see the model contract in `docs/data_format.md` |
| `ui/PoseOverlayView.kt` | A | Draws landmarks and the per rep marker |
| `MainActivity.kt`, navigation | B | Screen flow |
| `ui/StartScreen`, `ui/CountdownScreen`, `ui/RecordScreen`, `ui/ResultScreen`, `ui/LeaderboardScreen` | B | Screens, the record screen hosts A's camera view and overlay |
| `net/Config.kt` | B | `SERVER_URL`, the one constant to change at the demo |
| `net/ApiClient.kt` | B | `POST /runs` with retry, `GET /leaderboard`, `GET /model/latest` |
| `net/RunPayload.kt` | B | JSON payload from `docs/data_format.md` |
| `model/ModelStore.kt` | B | Saves the downloaded `.tflite` and gives its path to A's `ModelCounter` |
| `AndroidManifest.xml`, `res/xml/network_security_config.xml`, Gradle files | B | Coordinate with A before adding dependencies |

## The seam between A and B

Agree on these two types first and commit them together, then each side works alone. A's code produces frames and counts. B's code records the frames for upload and shows the count.

```kotlin
// pose/LandmarkFrame.kt
// lm holds 12 floats: x, y for the 6 landmarks in the order of docs/data_format.md
data class LandmarkFrame(val tMs: Long, val lm: FloatArray)

// counter/RepCounter.kt
interface RepCounter {
    val count: Int
    /** Feed one frame, returns true when this frame completes a rep. */
    fun onFrame(frame: LandmarkFrame): Boolean
    fun reset()
}
```

The rule based counter must work before the model, so the demo never depends on the model.

## Network setup for the demo

The server is plain HTTP on the laptop, so Android needs cleartext allowed. Add `android:networkSecurityConfig="@xml/network_security_config"` and the `INTERNET` permission to the manifest, then:

```xml
<!-- res/xml/network_security_config.xml -->
<network-security-config>
    <base-config cleartextTrafficPermitted="true" />
</network-security-config>
```

Phone and laptop must be on the same Wi-Fi. The laptop address goes in `net/Config.kt`.

## Milestones

1. Camera preview with pose landmarks drawn (A).
2. Start, countdown, and result screens with placeholder numbers (B).
3. Rule based count shown live with a marker per rep (A).
4. A recorded run uploads to the server and shows on the leaderboard (B). Team check: first end to end run.
5. Download the latest model and count with it (A and B).
