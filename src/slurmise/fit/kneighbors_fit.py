from __future__ import annotations

from dataclasses import InitVar, dataclass

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    def _make_model(self, categorical_features, numerical_features):
        categorical_transformer = Pipeline(
            steps=[
                ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist")),
            ]
        )
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=np.max)),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        model = KNeighborsRegressor(self.nneighbors)

        return Pipeline([("preprocessor", preprocessor), ("model", model)])
