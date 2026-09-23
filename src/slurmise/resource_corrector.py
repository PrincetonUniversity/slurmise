from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_VALID_UNCERTAINTY = ("default", "prediction", "max", "min")
_RESOURCE_BUILTIN_DEFAULTS: dict[str, float] = {"runtime": 60.0, "memory": 1000.0}

OnHighUncertainty = Literal["default", "prediction", "max", "min"]


@dataclass(frozen=True)
class ResourceCorrector:
    """Applies all post-prediction corrections for a single resource (runtime or memory).

    Constructed once per job per resource from merged global + job-level toml config.
    Call correct() during prediction to clamp, scale, and handle uncertainty.
    """

    resource: str
    default: float
    minimum: float = 0.0
    maximum: float = math.inf
    multiply_prediction_by: float = 1.0
    retry_exponent: float = 1.0
    on_high_uncertainty_return: OnHighUncertainty = "prediction"

    def __post_init__(self):
        if self.on_high_uncertainty_return not in _VALID_UNCERTAINTY:
            raise ValueError(
                f"on_high_uncertainty_return must be one of {_VALID_UNCERTAINTY!r}, "
                f"got {self.on_high_uncertainty_return!r}"
            )
        if self.minimum < 0:
            raise ValueError(f"minimum must be >= 0, got {self.minimum}")
        if self.maximum < self.minimum:
            raise ValueError(f"maximum ({self.maximum}) must be >= minimum ({self.minimum})")
        if self.multiply_prediction_by <= 0:
            raise ValueError(f"multiply_prediction_by must be > 0, got {self.multiply_prediction_by}")
        if self.on_high_uncertainty_return == "max" and not math.isfinite(self.maximum):
            raise ValueError("on_high_uncertainty_return='max' requires a finite maximum to be set")

    @classmethod
    def from_config(cls, resource: str, global_config: dict, job_config: dict | None = None) -> ResourceCorrector:
        """Merge global and per-job config dicts into a corrector.

        Job-level values override global; missing keys fall back to built-in defaults.
        """
        merged = {**global_config, **(job_config or {})}
        builtin_default = _RESOURCE_BUILTIN_DEFAULTS.get(resource, 0.0)
        return cls(
            resource=resource,
            default=float(merged.get("default", builtin_default)),
            minimum=float(merged.get("minimum", 0.0)),
            maximum=float(merged.get("maximum", math.inf)),
            multiply_prediction_by=float(merged.get("multiply_prediction_by", 1.0)),
            retry_exponent=float(merged.get("retry_exponent", 1.0)),
            on_high_uncertainty_return=merged.get("on_high_uncertainty_return", "prediction"),
        )

    def correct(
        self, predicted: float, is_high_uncertainty: bool, job_name: str, attempt: int = 0
    ) -> tuple[float, list[str]]:
        """Apply all corrections to a raw model prediction.

        Order of operations:
          1. Negative/zero check → return default
          2. Retry scaling: predicted *= attempt**retry_exponent  (only when attempt > 0)
          3. Static scaling: *= multiply_prediction_by
          4. High-uncertainty branch (on_high_uncertainty_return)
          5. Clamp to [minimum, maximum]
        """
        warnings: list[str] = []

        if predicted <= 0:
            fallback = self.on_high_uncertainty_return if self.on_high_uncertainty_return != "prediction" else "max"
            warnings.append(f"Predicted {self.resource} for job {job_name} is zero or negative ({predicted:.4g}).")
            match fallback:
                case "default":
                    return self.default, warnings
                case "max":
                    if math.isfinite(self.maximum):
                        return self.maximum, warnings
                    return self.default, warnings
                case "min":
                    return self.minimum, warnings

        if attempt > 0:
            predicted = predicted * attempt**self.retry_exponent

        scaled = predicted * self.multiply_prediction_by

        if is_high_uncertainty:
            match self.on_high_uncertainty_return:
                case "default":
                    warnings.append(
                        f"{self.resource.capitalize()} prediction for job {job_name} has high uncertainty. "
                        f"Returning default {self.resource} value."
                    )
                    return self.default, warnings
                case "max":
                    warnings.append(
                        f"{self.resource.capitalize()} prediction for job {job_name} has high uncertainty. "
                        f"Returning maximum {self.resource} value."
                    )
                    return self.maximum, warnings
                case "min":
                    warnings.append(
                        f"{self.resource.capitalize()} prediction for job {job_name} has high uncertainty. "
                        f"Returning minimum {self.resource} value."
                    )
                    return self.minimum, warnings
                case "prediction":
                    warnings.append(f"{self.resource.capitalize()} prediction for job {job_name} has high uncertainty.")

        if scaled > self.maximum:
            warnings.append(
                f"Predicted {self.resource} for job {job_name} ({scaled:.4g}) exceeds maximum "
                f"({self.maximum:.4g}), clamping."
            )
            scaled = self.maximum

        result = max(scaled, self.minimum)
        return result, warnings
