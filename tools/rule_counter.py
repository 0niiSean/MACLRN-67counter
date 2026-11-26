"""Reference rule based rep counter.

This is the single definition of the baseline counter that everyone shares:

* Member A ports it to Kotlin for the live count on the phone and tunes the
  thresholds (record the final values in the tuning notes and here).
* Member D runs it over exported runs to get per rep timestamps for training
  labels (keep only runs where its total matches the hand count).

If you change the logic or the default thresholds, change both this file and
the Kotlin port in the same pull request so the two never drift apart.

Signal
    s = (y_right_wrist - y_left_wrist) / shoulder_width

Image y grows downward, so s is "left wrist height minus right wrist height",
scaled by shoulder width so the threshold does not depend on distance.

Counting (hysteresis, works frame by frame on the phone)
    1. Smooth s with a moving average of `smooth` frames.
    2. The counter starts armed, so the first left hand top counts.
    3. Count a rep when s rises above +threshold (left hand clearly up) while
       armed, provided at least `min_gap_ms` passed since the last rep.
    4. Arm again when s drops below -threshold (right hand clearly up).

One rep is therefore each time the left hand goes to the top, as in the rep
definition in docs/data_format.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Sequence

# Landmark indices, see docs/data_format.md
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = range(6)

DEFAULT_THRESHOLD = 0.3
DEFAULT_SMOOTH = 3
DEFAULT_MIN_GAP_MS = 150


def wrist_signal(lm: Sequence[Sequence[float]]) -> float:
    """Signal for one frame. `lm` is 6 [x, y] pairs in the contract order."""
    shoulder_width = abs(lm[R_SHOULDER][0] - lm[L_SHOULDER][0])
    shoulder_width = max(shoulder_width, 1e-3)
    return (lm[R_WRIST][1] - lm[L_WRIST][1]) / shoulder_width


@dataclass
class RuleCounter:
    threshold: float = DEFAULT_THRESHOLD
    smooth: int = DEFAULT_SMOOTH
    min_gap_ms: int = DEFAULT_MIN_GAP_MS

    count: int = 0
    rep_end_ms: List[int] = field(default_factory=list)
    _window: Deque[float] = field(default_factory=deque)
    _armed: bool = True
    _last_rep_ms: int = -(10**9)

    def update(self, t_ms: int, lm: Sequence[Sequence[float]]) -> bool:
        """Feed one frame. Returns True when this frame completes a rep."""
        self._window.append(wrist_signal(lm))
        if len(self._window) > self.smooth:
            self._window.popleft()
        s = sum(self._window) / len(self._window)

        if s < -self.threshold:
            self._armed = True
        elif s > self.threshold and self._armed and t_ms - self._last_rep_ms >= self.min_gap_ms:
            self._armed = False
            self._last_rep_ms = t_ms
            self.count += 1
            self.rep_end_ms.append(t_ms)
            return True
        return False


def count_run(frames: Sequence[dict], **params) -> RuleCounter:
    """Run the counter over a list of {"t_ms", "lm"} frames from a payload."""
    counter = RuleCounter(**params)
    for f in frames:
        counter.update(f["t_ms"], f["lm"])
    return counter


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python tools/rule_counter.py <run_payload.json>")
    with open(sys.argv[1], encoding="utf-8") as fh:
        run = json.load(fh)
    c = count_run(run["frames"])
    print(f"rule count {c.count}  app {run.get('app_rep_count')}  label {run.get('label_rep_count')}")
