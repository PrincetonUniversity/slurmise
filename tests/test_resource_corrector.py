"""Tests for ResourceCorrector correctness and configuration merging."""

import math

import pytest

from slurmise.resource_corrector import ResourceCorrector


def make_corrector(**kwargs) -> ResourceCorrector:
    defaults = {"resource": "runtime", "default": 60.0}
    return ResourceCorrector(**{**defaults, **kwargs})


# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


def test_invalid_on_high_uncertainty_return():
    with pytest.raises(ValueError, match="on_high_uncertainty_return"):
        make_corrector(on_high_uncertainty_return="wrong")


def test_negative_minimum_rejected():
    with pytest.raises(ValueError, match="minimum"):
        make_corrector(minimum=-1)


def test_maximum_less_than_minimum_rejected():
    with pytest.raises(ValueError, match="maximum"):
        make_corrector(minimum=100, maximum=50)


def test_nonpositive_multiply_prediction_by_rejected():
    with pytest.raises(ValueError, match="multiply_prediction_by"):
        make_corrector(multiply_prediction_by=0)


def test_from_config_uses_builtin_runtime_default():
    c = ResourceCorrector.from_config("runtime", {})
    assert c.default == 60
    assert c.minimum == 0
    assert c.maximum == math.inf
    assert c.multiply_prediction_by == 1.0
    assert c.retry_exponent == 1.0
    assert c.on_high_uncertainty_return == "prediction"


def test_from_config_uses_builtin_memory_default():
    c = ResourceCorrector.from_config("memory", {})
    assert c.default == 1000


def test_from_config_global_overrides_builtin():
    c = ResourceCorrector.from_config("runtime", {"default": 120, "minimum": 5, "maximum": 1440})
    assert c.default == 120
    assert c.minimum == 5
    assert c.maximum == 1440


def test_from_config_job_overrides_global():
    global_cfg = {"default": 60, "minimum": 5, "maximum": 1440}
    job_cfg = {"default": 240, "maximum": 480}
    c = ResourceCorrector.from_config("runtime", global_cfg, job_cfg)
    assert c.default == 240
    assert c.minimum == 5  # not in job_cfg → inherited from global
    assert c.maximum == 480


def test_from_config_all_fields():
    cfg = {
        "default": 90,
        "minimum": 10,
        "maximum": 2000,
        "multiply_prediction_by": 1.5,
        "retry_exponent": 2.0,
        "on_high_uncertainty_return": "max",
    }
    c = ResourceCorrector.from_config("memory", cfg)
    assert c.default == 90
    assert c.minimum == 10
    assert c.maximum == 2000
    assert c.multiply_prediction_by == pytest.approx(1.5)
    assert c.retry_exponent == pytest.approx(2.0)
    assert c.on_high_uncertainty_return == "max"


# ---------------------------------------------------------------------------
# correct() – normal path (no uncertainty)
# ---------------------------------------------------------------------------


def test_correct_normal_returns_prediction():
    c = make_corrector(minimum=0, maximum=math.inf)
    value, warnings = c.correct(42.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(42.0)
    assert warnings == []


def test_correct_clamps_to_maximum():
    c = make_corrector(maximum=100)
    value, warnings = c.correct(500.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(100)
    assert any("maximum" in w for w in warnings)


def test_correct_clamps_to_minimum():
    c = make_corrector(minimum=50, maximum=math.inf, default=60)
    value, warnings = c.correct(10.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(50)
    assert warnings == []


def test_correct_zero_prediction_no_max_returns_default():
    """With on_high_uncertainty_return='prediction' and no max, negative → default."""
    c = make_corrector(default=60, on_high_uncertainty_return="prediction")
    value, warnings = c.correct(0.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(60)
    assert any("zero or negative" in w for w in warnings)


def test_correct_negative_prediction_no_max_returns_default():
    """With on_high_uncertainty_return='prediction' and no max, negative → default."""
    c = make_corrector(default=60, on_high_uncertainty_return="prediction")
    value, warnings = c.correct(-5.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(60)
    assert any("zero or negative" in w for w in warnings)


def test_correct_negative_prediction_returns_maximum_when_configured():
    """With on_high_uncertainty_return='prediction' and a max, negative → max (not prediction)."""
    c = make_corrector(default=60, maximum=1000, on_high_uncertainty_return="prediction")
    value, warnings = c.correct(-5.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(1000)
    assert any("zero or negative" in w for w in warnings)


def test_correct_negative_prediction_on_high_uncertainty_default_returns_default():
    """With on_high_uncertainty_return='default', negative → default regardless of max."""
    c = make_corrector(default=60, maximum=1000, on_high_uncertainty_return="default")
    value, warnings = c.correct(-5.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(60)
    assert any("zero or negative" in w for w in warnings)


def test_correct_negative_prediction_on_high_uncertainty_min_returns_minimum():
    """With on_high_uncertainty_return='min', negative → minimum."""
    c = make_corrector(default=60, minimum=10, on_high_uncertainty_return="min")
    value, warnings = c.correct(-5.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(10)
    assert any("zero or negative" in w for w in warnings)


def test_correct_multiply_prediction_by():
    c = make_corrector(multiply_prediction_by=2.0)
    value, warnings = c.correct(30.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(60.0)
    assert warnings == []


def test_correct_multiply_then_clamp():
    c = make_corrector(multiply_prediction_by=3.0, maximum=100)
    value, warnings = c.correct(50.0, is_high_uncertainty=False, job_name="job")
    assert value == pytest.approx(100)
    assert any("maximum" in w for w in warnings)


# ---------------------------------------------------------------------------
# correct() – retry scaling
# ---------------------------------------------------------------------------


def test_correct_attempt_zero_no_scaling():
    c = make_corrector(retry_exponent=2.0)
    value, _warnings = c.correct(50.0, is_high_uncertainty=False, job_name="job", attempt=0)
    assert value == pytest.approx(50.0)


def test_correct_attempt_one_linear_exponent():
    c = make_corrector(retry_exponent=1.0)
    value, _warnings = c.correct(50.0, is_high_uncertainty=False, job_name="job", attempt=1)
    assert value == pytest.approx(50.0)  # 50 * 1**1 = 50


def test_correct_attempt_two_linear_exponent():
    c = make_corrector(retry_exponent=1.0)
    value, _warnings = c.correct(50.0, is_high_uncertainty=False, job_name="job", attempt=2)
    assert value == pytest.approx(100.0)  # 50 * 2**1 = 100


def test_correct_attempt_two_quadratic_exponent():
    c = make_corrector(retry_exponent=2.0)
    value, _warnings = c.correct(50.0, is_high_uncertainty=False, job_name="job", attempt=2)
    assert value == pytest.approx(200.0)  # 50 * 2**2 = 200


def test_correct_retry_scaling_then_clamp():
    c = make_corrector(retry_exponent=2.0, maximum=100)
    value, warnings = c.correct(50.0, is_high_uncertainty=False, job_name="job", attempt=2)
    assert value == pytest.approx(100.0)  # 50 * 4 = 200, clamped to 100
    assert any("maximum" in w for w in warnings)


# ---------------------------------------------------------------------------
# correct() – high uncertainty branches
# ---------------------------------------------------------------------------


def test_high_uncertainty_prediction_returns_value():
    """on_high_uncertainty_return='prediction' (default) still returns the prediction."""
    c = make_corrector(on_high_uncertainty_return="prediction", default=60)
    value, warnings = c.correct(42.0, is_high_uncertainty=True, job_name="job")
    assert value == pytest.approx(42.0)
    assert any("uncertainty" in w for w in warnings)


def test_high_uncertainty_default_returns_default():
    c = make_corrector(on_high_uncertainty_return="default", default=60)
    value, warnings = c.correct(42.0, is_high_uncertainty=True, job_name="job")
    assert value == pytest.approx(60)
    assert any("default" in w for w in warnings)


def test_high_uncertainty_max_returns_maximum():
    c = make_corrector(on_high_uncertainty_return="max", default=60, maximum=200)
    value, warnings = c.correct(42.0, is_high_uncertainty=True, job_name="job")
    assert value == pytest.approx(200)
    assert any("maximum" in w for w in warnings)


def test_high_uncertainty_max_no_maximum_raises():
    """on_high_uncertainty_return='max' requires a finite maximum at construction time."""
    with pytest.raises(ValueError, match="finite maximum"):
        make_corrector(on_high_uncertainty_return="max", default=60, maximum=math.inf)


def test_high_uncertainty_min_returns_minimum():
    c = make_corrector(on_high_uncertainty_return="min", minimum=10)
    value, warnings = c.correct(42.0, is_high_uncertainty=True, job_name="job")
    assert value == pytest.approx(10)
    assert any("minimum" in w for w in warnings)


def test_high_uncertainty_prediction_still_clamped():
    """Even with on_high_uncertainty_return='prediction', the value is clamped to [min, max]."""
    c = make_corrector(on_high_uncertainty_return="prediction", maximum=30)
    value, _warnings = c.correct(42.0, is_high_uncertainty=True, job_name="job")
    assert value == pytest.approx(30)
