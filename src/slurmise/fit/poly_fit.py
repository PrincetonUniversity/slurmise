from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import ClassVar

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from slurmise.fit.resource_fit import ResourceFit
from slurmise.job_data import JobData


@dataclass(kw_only=True)
class PolynomialFit(ResourceFit):
    degree: int = 2
    runtime_model: InitVar[Pipeline | None] = None
    memory_model: InitVar[Pipeline | None] = None
    _runtime_model_name: ClassVar[str] = "poly_runtime_model.pkl"
    _memory_model_name: ClassVar[str] = "poly_memory_model.pkl"

    def __post_init__(self, runtime_model, memory_model):
        self.runtime_model = runtime_model
        self.memory_model = memory_model

        super().__post_init__()

    def save(self):
        super().save()

    @classmethod
    def load(cls, query: JobData | None = None, path: str | None = None) -> PolynomialFit:
        fit_obj = super().load(query=query, path=path, degree=2)

        runtime_model = fit_obj.path / PolynomialFit._runtime_model_name
        fit_obj.runtime_model = joblib.load(str(runtime_model)) if runtime_model.exists() else None

        memory_model = fit_obj.path / PolynomialFit._memory_model_name
        fit_obj.memory_model = joblib.load(str(memory_model)) if memory_model.exists() else None

        return fit_obj

    def _make_model(self, categories, numerics) -> Pipeline:
        preprocessor = self._get_preprocessor(categories=categories, numerics=numerics)
        # We are doing polynomial regression, so we need to add polynomial features
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        model = LinearRegression()

        return Pipeline([("preprocessor", preprocessor), ("poly", poly), ("model", model)])

    def _design(self, model, X) -> np.ndarray:
        F = model[:-1].transform(X)
        F = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
        return np.column_stack([np.ones(len(F)), F])

    def _fit_uncertainty(self, model, X_train, y_train, X_test, y_test) -> None:
        F = self._design(model, X_train)
        n, p = F.shape
        if n <= p:
            return
        resid = np.asarray(y_train, dtype=float) - model.predict(X_train)
        model.slurmise_s2_ = float(np.sum(np.square(resid)) / (n - p))
        model.slurmise_xtx_inv_ = np.linalg.pinv(np.matmul(F.T, F))

    def _predict_std(self, model, X) -> float | None:
        s2 = getattr(model, "slurmise_s2_", None)
        xtx_inv = getattr(model, "slurmise_xtx_inv_", None)
        if s2 is None or xtx_inv is None:
            return None
        f0 = self._design(model, X)[0]
        return float(np.sqrt(s2 * (1 + np.linalg.multi_dot([f0, xtx_inv, f0]))))
