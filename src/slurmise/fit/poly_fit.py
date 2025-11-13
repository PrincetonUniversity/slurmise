from __future__ import annotations

from dataclasses import ClassVar, InitVar, dataclass

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from slurmise.fit.resource_fit import ResourceFit
from slurmise.job_data import JobData


@dataclass(kw_only=True)
class PolynomialFit(ResourceFit):
    degree: int
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

    def _make_model(self, categorical_features, numerical_features) -> Pipeline:

        preprocessor = self._get_preprocessor(
            categorical_features=categorical_features, numerical_features=numerical_features
        )
        # We are doing polynomial regression, so we need to add polynomial features
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        model = LinearRegression()

        return Pipeline([("preprocessor", preprocessor), ("poly", poly), ("model", model)])
