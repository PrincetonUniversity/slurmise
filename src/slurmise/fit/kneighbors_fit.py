from __future__ import annotations

from dataclasses import InitVar, dataclass

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

    def __post_init__(self, runtime_model, memory_model):
        self.runtime_model = runtime_model
        self.memory_model = memory_model

        super().__post_init__()

    def save(self):
        super().save()
        if self.runtime_model is not None:
            modelpath = self.path / "runtime_model.pkl"
            joblib.dump(self.runtime_model, str(modelpath))
        if self.memory_model is not None:
            modelpath = self.path / "memory_model.pkl"
            joblib.dump(self.memory_model, str(modelpath))

    @classmethod
    def load(cls, query: JobData | None = None, path: str | None = None) -> KNNFit:
        fit_obj = super().load(query=query, path=path, nneighbors=5)

        runtime_model = fit_obj.path / "runtime_model.pkl"
        fit_obj.runtime_model = joblib.load(str(runtime_model)) if runtime_model.exists() else None

        memory_model = fit_obj.path / "memory_model.pkl"
        fit_obj.memory_model = joblib.load(str(memory_model)) if memory_model.exists() else None

        return fit_obj

    def _make_model(self, categorical_features, numerical_features) -> Pipeline:

        preprocessor = self._get_preprocessor(categorical_features=categorical_features,
                                              numerical_features=numerical_features)
        model = KNeighborsRegressor(self.nneighbors)

        return Pipeline([("preprocessor", preprocessor), ("model", model)])
