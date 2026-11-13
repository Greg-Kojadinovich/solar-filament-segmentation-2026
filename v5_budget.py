"""Cooperative V5 training budget; reserve time for scoring and CSV export."""

import time


def remaining_training_seconds(started, total_seconds=7200, reserve_seconds=1200, now=None):
    if total_seconds <= reserve_seconds or reserve_seconds < 0:
        raise ValueError("Total runtime must exceed a nonnegative scoring/export reserve.")
    current = time.monotonic() if now is None else now
    return max(0.0, started + total_seconds - reserve_seconds - current)
