from dataclasses import InitVar, dataclass

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Generate a polynomial fit for the runtime data using sklearn
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from slurmise.job_data import JobData
from slurmise.utils import jobs_to_pandas
from slurmise.fit.resource_fit import ResourceFit


@dataclass(kw_only=True)
class PolynomialFit(ResourceFit):
    degree: int
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
    def load(cls, query: JobData | None = None, path: str | None = None)-> "PolynomialFit":
        fit_obj = super().load(query=query, path=path, degree=2)

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

        # We are doing polynomial regression, so we need to add polynomial features
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        model = LinearRegression()

        pipeline = Pipeline(
            [("preprocessor", preprocessor), ("poly", poly), ("model", model)]
        )

        return pipeline

    def fit(
        self, jobs: list[JobData], random_state: np.random.RandomState | None, **kwargs
    ):
        X, categorical_features, numerical_features = jobs_to_pandas(jobs)

        Y = X[["runtime", "memory"]]

        # Drop the runtime and memory columns
        X = X.drop(columns=["runtime", "memory"])

        # Split test and train data
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=random_state
        )

        self.last_fit_dsize = len(X_train)

        self.runtime_model = self._make_model(categorical_features, numerical_features)
        self.memory_model = self._make_model(categorical_features, numerical_features)

        self.runtime_model.fit(X_train, y_train["runtime"])
        self.memory_model.fit(X_train, y_train["memory"])

        # Evaluate the model on test
        Y_pred_runtime = self.runtime_model.predict(X_test)
        Y_pred_memory = self.memory_model.predict(X_test)

        def mean_percent_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        self.model_metrics = {
            "runtime": {
                "mpe": mean_percent_error(y_test["runtime"], Y_pred_runtime),
                "mse": mean_squared_error(y_test["runtime"], Y_pred_runtime),
            },
            "memory": {
                "mpe": mean_percent_error(y_test["memory"], Y_pred_memory),
                "mse": mean_squared_error(y_test["memory"], Y_pred_memory),
            },
        }
        # TODO: Warning if model metrics are larger than a threshold.

    def predict(self, job: JobData) -> tuple[JobData, list[str]]:
        # TODO: check if it can be abstracted.

        if self.last_fit_dsize < 10:

            return (
                job,
                "Not enough fitting data points in the fits. Returning default values.",
            )

        X, _, _ = jobs_to_pandas([job])
        warnmsg = []
        if self.model_metrics["runtime"]["mpe"] < 10:
            warnmsg += [
                f"Runtime prediction for job {job.job_name} is not within 10% of actual value.",
                "Returing default runtime value.",
            ]
        else:
            predicted_runtime = self.runtime_model.predict(X)[0]
            if predicted_runtime > 0 and predicted_runtime < 100 * job.runtime:
                job.runtime = predicted_runtime
            else:
                warnmsg += [
                    f"Predicted runtime for job {job.job_name} is either negative or more than 100 times larger than default.",
                    "Returing default runtime value.",
                ]

        if self.model_metrics["memory"]["mpe"] < 10:
            warnmsg += [
                f"Memory prediction for job {job.job_name} is not within 10% of actual value.",
                "Returing default memory value.",
            ]
        else:
            predicted_memory = self.memory_model.predict(X)[0]
            if predicted_memory > 0 and predicted_memory < 100 * job.memory:
                job.memory = predicted_memory
            else:
                warnmsg += [
                    f"Predicted memory for job {job.job_name} is either negative or more than 100 times larger than default.",
                    "Returing default memory value.",
                ]

        return job, warnmsg
