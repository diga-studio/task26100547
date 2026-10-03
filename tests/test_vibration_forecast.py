"""Tests for the vibration_forecast library."""

import math

import pytest

from vibration_forecast import (
    URGENT_THRESHOLD_S,
    WINDOW_US,
    Forecast,
    VibrationTrendForecaster,
)

US = 1_000_000
# Large base timestamp so window arithmetic stays far from zero.
T0 = 20_000_000_000_000


def _feed(f, samples):
    for t, v in samples:
        f.append(t, v)


class TestClockAndValidation:
    def test_empty_and_single_sample_give_empty_forecast(self):
        f = VibrationTrendForecaster(100.0)
        assert f.forecast(T0) == Forecast(seconds=None, urgent=False)
        f.append(T0 + US, 1.0)
        assert f.forecast(T0 + US) == Forecast(seconds=None, urgent=False)

    def test_sample_timestamps_must_be_strictly_increasing(self):
        f = VibrationTrendForecaster(100.0)
        f.append(T0 + 2 * US, 1.0)
        with pytest.raises(ValueError):
            f.append(T0 + 2 * US, 2.0)  # equal
        with pytest.raises(ValueError):
            f.append(T0 + US, 2.0)  # backward

    def test_forecast_cannot_move_clock_backward(self):
        f = VibrationTrendForecaster(100.0)
        f.append(T0 + 5 * US, 1.0)
        with pytest.raises(ValueError):
            f.forecast(T0 + 4 * US)
        with pytest.raises(ValueError):
            f.append(T0 + 4 * US, 2.0)

    def test_append_at_clock_is_allowed_after_forecast(self):
        f = VibrationTrendForecaster(100.0)
        f.append(T0 + US, 1.0)
        f.forecast(T0 + 5 * US)
        f.append(T0 + 5 * US, 2.0)  # equal to clock, but a new sample time
        assert f.forecast(T0 + 5 * US).seconds is not None

    def test_rejects_non_finite_values_and_bad_types(self):
        f = VibrationTrendForecaster(100.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                f.append(T0 + US, bad)
        with pytest.raises(TypeError):
            f.append(1.5, 1.0)
        with pytest.raises(TypeError):
            f.append(True, 1.0)
        with pytest.raises(ValueError):
            VibrationTrendForecaster(float("nan"))

    def test_fitted_at_or_above_threshold_gives_zero_and_urgent(self):
        f = VibrationTrendForecaster(10.0)
        _feed(f, [(T0 + US, 9.0), (T0 + 2 * US, 11.0)])
        assert f.forecast(T0 + 2 * US) == Forecast(seconds=0.0, urgent=True)

    def test_negative_and_zero_slope_give_empty_forecast(self):
        f = VibrationTrendForecaster(100.0)
        _feed(f, [(T0 + US, 12.0), (T0 + 2 * US, 8.0)])
        assert f.forecast(T0 + 2 * US) == Forecast(seconds=None, urgent=False)
        g = VibrationTrendForecaster(100.0)
        _feed(g, [(T0 + US, 5.0), (T0 + 2 * US, 5.0)])
        assert g.forecast(T0 + 2 * US) == Forecast(seconds=None, urgent=False)

    def test_append_also_evicts_expired_samples(self):
        f = VibrationTrendForecaster(100.0)
        f.append(T0 + US, 1.0)
        f.append(T0 + 2 * US, 2.0)
        f.append(T0 + 2 * US + WINDOW_US + 1, 3.0)  # first two now expired
        assert f.forecast(T0 + 2 * US + WINDOW_US + 1) == Forecast(None, False)


class TestShiftInvarianceAndUrgencyBoundary:
    # Irregular whole-second offsets (microsecond-aligned), all inside window.
    OFFSETS_S = (-604_799, -401_233, -123_456, -1)
    SLOPE = 1.0 / 1024.0  # exact in binary64
    H = 1000.0

    def _samples(self, now, fitted_now):
        return [
            (now + off * US, fitted_now + off * self.SLOPE)
            for off in self.OFFSETS_S
        ]

    @pytest.mark.parametrize("shift", [0, 777_777_777_777, 9 * 3600 * US])
    def test_prediction_exactly_72h_is_not_urgent(self, shift):
        # fitted(now) = H - 253.125 -> (H - fitted) / slope == 259200 exactly.
        now = T0 + shift + 700_000 * US
        f = VibrationTrendForecaster(self.H)
        _feed(f, self._samples(now, 746.875))
        result = f.forecast(now)
        assert result.seconds == URGENT_THRESHOLD_S == 259200.0
        assert result.urgent is False

    @pytest.mark.parametrize("shift", [0, 777_777_777_777, 9 * 3600 * US])
    def test_prediction_just_below_72h_is_urgent(self, shift):
        # fitted(now) = 746.875 + 1/2048 -> seconds == 259199.5 exactly.
        now = T0 + shift + 700_000 * US
        f = VibrationTrendForecaster(self.H)
        _feed(f, self._samples(now, 746.875 + 1.0 / 2048.0))
        result = f.forecast(now)
        assert result.seconds == 259199.5 < URGENT_THRESHOLD_S
        assert result.urgent is True

    def test_shift_leaves_forecast_bit_identical(self):
        results = []
        for shift in (0, 123_456_789_012, 51 * 3600 * US):
            now = T0 + shift + 700_000 * US
            f = VibrationTrendForecaster(self.H)
            _feed(f, self._samples(now, 700.0))
            results.append(f.forecast(now))
        assert results[0] == results[1] == results[2]
        assert results[0].seconds == pytest.approx(300.0 * 1024.0)


class TestSpikeExpiry:
    NOW2_OFFSET = WINDOW_US  # now2 = t_spike + WINDOW_US puts spike on the edge

    def _build(self, tail_values):
        f = VibrationTrendForecaster(100.0)
        f.append(T0, 1_000_000.0)  # huge spike
        for k, v in enumerate(tail_values, start=1):
            f.append(T0 + k * US, v)
        return f

    def test_spike_live_then_expires_at_left_edge_constant_tail(self):
        tail = [10.0] * 6
        f = self._build(tail)

        # Spike exactly on the left-open boundary: expired.
        got = f.forecast(T0 + self.NOW2_OFFSET)

        fresh = VibrationTrendForecaster(100.0)
        _feed(fresh, [(T0 + k * US, v) for k, v in enumerate(tail, start=1)])
        expected = fresh.forecast(T0 + self.NOW2_OFFSET)

        assert got == expected == Forecast(seconds=None, urgent=False)

    def test_spike_live_then_expires_at_left_edge_tiny_uptrend(self):
        tail = [10.0 + k * 1e-6 for k in range(1, 7)]  # slope 1e-6/s
        f = self._build(tail)

        # One second before the spike reaches the edge it is still live and
        # dominates the fit (huge negative slope -> no prediction).
        live = f.forecast(T0 + self.NOW2_OFFSET - US)
        assert live == Forecast(seconds=None, urgent=False)

        got = f.forecast(T0 + self.NOW2_OFFSET)

        fresh = VibrationTrendForecaster(100.0)
        _feed(fresh, [(T0 + k * US, v) for k, v in enumerate(tail, start=1)])
        expected = fresh.forecast(T0 + self.NOW2_OFFSET)

        assert got == expected  # no residual false trend from the spike
        assert got != live  # expiry actually changed the answer
        assert got.seconds is not None and got.seconds > 1e7
        assert got.urgent is False

    def test_window_empties_then_stream_recovers(self):
        f = self._build([10.0] * 6)
        now_empty = T0 + 6 * US + WINDOW_US  # last sample exactly on the edge
        assert f.forecast(now_empty) == Forecast(seconds=None, urgent=False)

        # Stream recovers: new samples after the empty window work normally.
        f.append(now_empty + US, 50.0)
        f.append(now_empty + 2 * US, 60.0)
        result = f.forecast(now_empty + 2 * US)
        assert result.seconds == pytest.approx(4.0)  # slope 10/s, 40 to go
        assert result.urgent is True


class TestOlsSemantics:
    def test_forecast_starts_from_fitted_line_not_last_reading(self):
        # Noisy data whose last raw reading is far below the fitted value.
        f = VibrationTrendForecaster(100.0)
        samples = [
            (T0 + 1 * US, 10.0),
            (T0 + 2 * US, 30.0),
            (T0 + 3 * US, 50.0),
            (T0 + 4 * US, 70.0),
            (T0 + 5 * US, 10.0),  # dip: raw last reading is low
        ]
        _feed(f, samples)
        now = T0 + 5 * US
        result = f.forecast(now)
        # OLS fit: slope 4/s, fitted value at now is 42.0 -- not the raw
        # last reading of 10.0.
        assert result.seconds == pytest.approx((100.0 - 42.0) / 4.0)

    def test_uses_only_samples_inside_window(self):
        f = VibrationTrendForecaster(100.0)
        f.append(T0, 95.0)  # would change the fit if still counted
        f.append(T0 + WINDOW_US + US, 1.0)
        f.append(T0 + WINDOW_US + 2 * US, 2.0)
        result = f.forecast(T0 + WINDOW_US + 2 * US)
        assert result.seconds == pytest.approx(98.0)
