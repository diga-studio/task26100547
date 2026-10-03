"""In-process 7-day vibration trend forecasting via ordinary least squares."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

WINDOW_US: int = 7 * 24 * 3600 * 1_000_000
URGENT_THRESHOLD_S: float = 259200.0
_US_PER_S: float = 1_000_000.0


@dataclass(frozen=True)
class Forecast:
    """Result of a forecast query.

    seconds: predicted seconds until the fitted trend reaches the threshold,
             or None when no such prediction exists.
    urgent:  True only when a prediction exists and is strictly below 259200 s.
    """

    seconds: Optional[float]
    urgent: bool


class VibrationTrendForecaster:
    """Tracks one sensor stream and predicts when readings reach threshold H.

    The effective window is the most recent 7 days, left-open and
    right-closed: a sample at exactly ``clock - WINDOW_US`` is expired.
    Both operations advance a single logical clock that never goes back;
    sample timestamps must be strictly increasing.
    """

    def __init__(self, threshold: float) -> None:
        h = float(threshold)
        if not math.isfinite(h):
            raise ValueError("threshold must be a finite number")
        self._h = h
        self._samples: List[Tuple[int, float]] = []
        self._clock: Optional[int] = None
        self._last_sample_t: Optional[int] = None

    def append(self, t: int, value: float) -> None:
        """Add a sample at integer microsecond timestamp t, evicting expired ones."""
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("t must be an integer microsecond timestamp")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("value must be a finite binary64 reading")
        if self._last_sample_t is not None and t <= self._last_sample_t:
            raise ValueError("sample timestamps must be strictly increasing")
        if self._clock is not None and t < self._clock:
            raise ValueError("logical clock must not go backward")
        self._clock = t
        self._last_sample_t = t
        self._evict()
        self._samples.append((t, v))

    def forecast(self, now: int) -> Forecast:
        """Advance the logical clock to now and return the maintenance forecast."""
        if isinstance(now, bool) or not isinstance(now, int):
            raise TypeError("now must be an integer microsecond timestamp")
        if self._clock is not None and now < self._clock:
            raise ValueError("logical clock must not go backward")
        self._clock = now
        self._evict()
        n = len(self._samples)
        if n < 2:
            return Forecast(seconds=None, urgent=False)

        # OLS on times relative to now (seconds) for numerical stability.
        xs = [(t - now) / _US_PER_S for t, _ in self._samples]
        ys = [v for _, v in self._samples]
        x_mean = math.fsum(xs) / n
        y_mean = math.fsum(ys) / n
        sxx = math.fsum((x - x_mean) ** 2 for x in xs)
        sxy = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        slope = sxy / sxx  # per second; sxx > 0 since timestamps are distinct
        fitted_now = y_mean - slope * x_mean  # fitted value at x = 0 (now)

        if fitted_now >= self._h:
            return Forecast(seconds=0.0, urgent=True)
        if slope <= 0.0:
            return Forecast(seconds=None, urgent=False)
        seconds = (self._h - fitted_now) / slope
        return Forecast(seconds=seconds, urgent=seconds < URGENT_THRESHOLD_S)

    def _evict(self) -> None:
        cutoff = self._clock - WINDOW_US
        self._samples = [s for s in self._samples if s[0] > cutoff]
