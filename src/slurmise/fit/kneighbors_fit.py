from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import ClassVar

import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from slurmise.fit.resource_fit import ResourceFit
from slurmise.job_data import JobData


@dataclass(kw_only=True)
class KNNFit(ResourceFit):
    nneighbors: int = 5
    runtime_model: InitVar[Pipeline | None] = None
    memory_model: InitVar[Pipeline | None] = None
    _runtime_model_name: ClassVar[str] = "knn_runtime_model.pkl"
    _memory_model_name: ClassVar[str] = "knn_memory_model.pkl"

    def __post_init__(self, runtime_model, memory_model):
        self.runtime_model = runtime_model
        self.memory_model = memory_model

        super().__post_init__()

    @classmethod
    def load(cls, query: JobData | None = None, path: str | None = None) -> KNNFit:
        fit_obj = super().load(query=query, path=path, nneighbors=5)

        runtime_model = fit_obj.path / KNNFit._runtime_model_name
        fit_obj.runtime_model = joblib.load(str(runtime_model)) if runtime_model.exists() else None

        memory_model = fit_obj.path / KNNFit._memory_model_name
        fit_obj.memory_model = joblib.load(str(memory_model)) if memory_model.exists() else None

        return fit_obj

    def _make_model(self, categories, numerics) -> Pipeline:
        preprocessor = self._get_preprocessor(categories=categories, numerics=numerics)
        model = KNeighborsRegressor(self.nneighbors)

        return Pipeline([("preprocessor", preprocessor), ("model", model)])
