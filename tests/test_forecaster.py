import math

import pytest

from vibration_trend import Forecast, TrendForecaster

SECOND = 1_000_000
DAY = 86_400 * SECOND
WINDOW = 7 * DAY
URGENT = 259_200  # 72h in seconds


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_fewer_than_two_points_gives_empty_forecast():
    f = TrendForecaster(threshold=10.0)
    assert f.forecast(0) == Forecast(seconds=None, urgent=False)
    f.append(0, 1.0)
    assert f.forecast(SECOND) == Forecast(seconds=None, urgent=False)


def test_fitted_value_at_or_above_threshold_predicts_zero_and_is_urgent():
    f = TrendForecaster(threshold=2.0)
    f.append(0, 2.0)
    f.append(SECOND, 3.0)  # fitted value at now is 3.0 >= H
    assert f.forecast(SECOND) == Forecast(seconds=0.0, urgent=True)


def test_zero_or_negative_slope_gives_empty_forecast():
    flat = TrendForecaster(threshold=10.0)
    flat.append(0, 1.0)
    flat.append(SECOND, 1.0)
    assert flat.forecast(SECOND) == Forecast(seconds=None, urgent=False)

    falling = TrendForecaster(threshold=10.0)
    falling.append(0, 5.0)
    falling.append(SECOND, 4.0)
    assert falling.forecast(SECOND) == Forecast(seconds=None, urgent=False)


def test_forecast_starts_from_fitted_line_not_last_reading():
    # Noisy data: last raw reading is far below the fitted line.
    f = TrendForecaster(threshold=10.0)
    f.append(0, 0.0)
    f.append(SECOND, 2.0)
    f.append(2 * SECOND, 1.0)
    result = f.forecast(2 * SECOND)
    # OLS through (0,0),(1,2),(2,1): slope 0.5/s, fitted at now = 1.5.
    assert result.seconds == pytest.approx((10.0 - 1.5) / 0.5)
    assert result.urgent


# ---------------------------------------------------------------------------
# Input validation and clock rules
# ---------------------------------------------------------------------------

def test_rejects_non_finite_or_non_float_values():
    f = TrendForecaster(threshold=10.0)
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            f.append(0, bad)
    with pytest.raises(TypeError):
        f.append(0, 1)  # int is not binary64
    with pytest.raises(TypeError):
        f.append(0, True)


def test_rejects_non_finite_threshold():
    with pytest.raises(ValueError):
        TrendForecaster(threshold=math.inf)
    with pytest.raises(TypeError):
        TrendForecaster(threshold=10)


def test_sample_timestamps_must_be_strictly_increasing():
    f = TrendForecaster(threshold=10.0)
    f.append(5, 1.0)
    with pytest.raises(ValueError):
        f.append(5, 2.0)
    with pytest.raises(ValueError):
        f.append(4, 2.0)


def test_logical_clock_never_goes_backwards():
    f = TrendForecaster(threshold=10.0)
    f.append(10, 1.0)
    with pytest.raises(ValueError):
        f.forecast(9)
    with pytest.raises(ValueError):
        f.append(9, 2.0)
    assert f.forecast(10) == Forecast(seconds=None, urgent=False)  # equal is fine


# ---------------------------------------------------------------------------
# Translation invariance under irregular sampling, 72h boundary
# ---------------------------------------------------------------------------

# Samples lie exactly on v = 5 + 2**-18 * (t_seconds - 3), so the OLS fit is
# exact in binary64. At now = 7s the fitted value is 5 + 2**-16 and the
# threshold sits exactly 2025/2048 = 259200 * 2**-18 above it, making the
# predicted horizon exactly 259200 seconds (72h) with zero float error.
_SLOPE = 2.0**-18
_BASE_THRESHOLD = 5.0 + 2.0**-16 + 2025.0 / 2048.0
_IRREGULAR_SECONDS = (0, 2, 3, 7)


def _run_72h_scenario(offset, threshold_adjust=0.0):
    f = TrendForecaster(threshold=_BASE_THRESHOLD + threshold_adjust)
    for t_sec in _IRREGULAR_SECONDS:
        f.append(offset + t_sec * SECOND, 5.0 + (t_sec - 3) * _SLOPE)
    return f.forecast(offset + 7 * SECOND)


def test_forecast_exactly_72h_is_not_urgent():
    result = _run_72h_scenario(offset=0)
    assert result.seconds == float(URGENT)
    assert result.urgent is False


def test_forecast_just_below_72h_is_urgent():
    result = _run_72h_scenario(offset=0, threshold_adjust=-_SLOPE)
    assert result.seconds == float(URGENT - 1)
    assert result.urgent is True


def test_forecast_above_72h_is_not_urgent():
    result = _run_72h_scenario(offset=0, threshold_adjust=2.0**-10)
    assert result.seconds == float(URGENT + 256)  # 2**-10 / 2**-18 == 256
    assert result.urgent is False


@pytest.mark.parametrize("adjust", [0.0, -_SLOPE, 2.0**-10])
def test_timestamp_translation_does_not_change_forecast(adjust):
    near = _run_72h_scenario(offset=0, threshold_adjust=adjust)
    far = _run_72h_scenario(offset=10**15, threshold_adjust=adjust)
    assert near == far


# ---------------------------------------------------------------------------
# Spike expiry at the open left boundary of the 7-day window
# ---------------------------------------------------------------------------

def _build_spike_stream(with_spike, trend_per_day):
    f = TrendForecaster(threshold=10.0)
    if with_spike:
        f.append(0, 1_000_000.0)
    for day in range(1, 7):
        f.append(day * DAY, 1.0 + trend_per_day * day)
    return f


@pytest.mark.parametrize("trend_per_day", [0.0, 1e-3])
def test_spike_valid_before_left_boundary_and_expired_at_it(trend_per_day):
    spiked = _build_spike_stream(with_spike=True, trend_per_day=trend_per_day)
    reference = _build_spike_stream(with_spike=False, trend_per_day=trend_per_day)

    # One microsecond before the boundary the spike is still inside the
    # window. With an underlying uptrend its distortion is observable: the
    # spike drags the fitted slope negative, so no forecast is produced
    # while the spike-free reference already predicts one.
    before = spiked.forecast(WINDOW - 1)
    if trend_per_day > 0.0:
        assert before == Forecast(seconds=None, urgent=False)
        assert before != reference.forecast(WINDOW - 1)
    else:
        assert before == Forecast(seconds=None, urgent=False)

    # At the boundary the spike's timestamp equals now - 7d; the window is
    # left-open, so it expires and no phantom trend may remain.
    after = spiked.forecast(WINDOW)
    assert after == reference.forecast(WINDOW)
    if trend_per_day == 0.0:
        assert after == Forecast(seconds=None, urgent=False)
    else:
        assert after.seconds is not None and after.seconds > 0.0


def test_window_empties_and_stream_recovers():
    f = _build_spike_stream(with_spike=True, trend_per_day=1e-3)
    f.forecast(WINDOW)

    # Advance far past every sample: the window is empty again.
    assert f.forecast(20 * DAY) == Forecast(seconds=None, urgent=False)

    # New samples after the empty window are forecast normally.
    f.append(20 * DAY, 2.0)
    f.append(21 * DAY, 3.0)  # slope: 1.0 per day
    result = f.forecast(21 * DAY)
    assert result.seconds == pytest.approx(7 * 86_400.0)
    assert result.urgent is False
