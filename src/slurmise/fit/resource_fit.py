from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass, field

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from slurmise.job_data import JobData
from slurmise.utils import jobs_to_pandas

BASEMODELPATH = pathlib.Path.home() / ".slurmise/models/"

# Above this mean percent error a model's prediction is reported with a warning.
MPE_THRESHOLD = 10
# Predictions at or beyond this multiple of the default are rejected outright.
MAX_PREDICTION_FACTOR = 100


@dataclass(kw_only=True)
class ResourceFit:
    query: JobData
    last_fit_dsize: int = 0
    fit_timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    model_metrics: dict = field(default_factory=dict)
    path: pathlib.Path | None = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = pathlib.Path(self.path)
        elif isinstance(self.path, pathlib.Path):
            pass
        elif self.path is None:
            self.path = self._make_model_path(query=self.query)
        else:
            raise ValueError("path must be a string or pathlib.Path object")

    @classmethod
    def _get_model_info_hash(cls, query: JobData) -> str:
        """
        This method generates a hash of the model's query information.
        """
        hash_info = {
            "class": cls.__name__,
            "job_name": query.job_name,
            **query.categories,
        }
        hash_info_tuple = tuple(hash_info.items())

        # Get and MD5 hash of information
        return hashlib.md5(str(hash_info_tuple).encode("utf-8")).hexdigest()

    @classmethod
    def _make_model_path(cls, query) -> pathlib.Path:
        """
        This method returns the path to the model's directory.

        The model's path is a function of the model's type and the hash of its query.
        """
        hash_val = cls._get_model_info_hash(query)
        return pathlib.Path(BASEMODELPATH) / cls.__name__ / hash_val

    def save(self, model_params: dict | None = None):
        """This method saves the basic information of the model, such as its query,
        when it was last fit, the dataset size of the latest fit, and the type of
        the model.
        """
        if model_params is None:
            model_params = {}

        self.path.mkdir(parents=True, exist_ok=True)
        with open(str(self.path / "fits.json"), "w") as save_file:
            # This converts the dataclass to a dictionary. If it is called from a subclass,
            # the subclass's attributes will be included in the dictionary.
            info = asdict(self)

            # Convert path to string
            info["path"] = str(info["path"])

            # Convert datetime to string
            info["fit_timestamp"] = info["fit_timestamp"].isoformat()

            info.update(model_params)
            json.dump(info, save_file)

        if self.runtime_model is not None:
            modelpath = self.path / self._runtime_model_name
            joblib.dump(self.runtime_model, str(modelpath))
        if self.memory_model is not None:
            modelpath = self.path / self._memory_model_name
            joblib.dump(self.memory_model, str(modelpath))

    @classmethod
    def load(cls, query: JobData | None = None, path: str | None = None, **kwargs) -> ResourceFit:
        """
        This method loads a model from a file. The model is loaded from the path
        provided, or from the path generated from the query.

        :param query: The query used to generate the model
        :type query: JobData
        :param path: The path to the model
        :type path: str
        :param kwargs: Additional keyword arguments to pass to the model
        :return: The model
        :rtype: ResourceFit

        """
        match [path, query]:
            case (None, None):
                raise ValueError("Either query or path must be provided")
            case (None, _):
                path = cls._make_model_path(query)
            case (str(path), _):
                path = pathlib.Path(path)

        if (path / "fits.json").exists():
            with open(str(path / "fits.json")) as load_file:
                info = json.load(load_file)

            # Convert datetime from isoformat string to datetime object
            info["fit_timestamp"] = datetime.datetime.fromisoformat(info["fit_timestamp"])
        else:
            info = {
                "query": query,
                "last_fit_dsize": 0,
                "fit_timestamp": datetime.datetime.now(tz=datetime.UTC),
                "model_metrics": {},
                "path": path,
            }
            info.update(kwargs)

        # Generates an instance of the class from the dictionary. When this is called by
        # a ResourceFit subclass, it includes all attributes of the subclass(es) and the
        # ResourceFit class.

        return cls(**info)

    @classmethod
    def mean_percent_error(cls, y_true, y_pred) -> float:
        """Mean percent error, skipping records whose true value is zero.

        Jobs that never ran are recorded with a zero runtime or memory, and dividing by
        those makes the error infinite for any database containing one.
        """
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        nonzero = y_true != 0
        return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    def _get_preprocessor(self, categories, numerics):
        category_transformer = Pipeline(
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
                ("num", numeric_transformer, numerics),
                ("cat", category_transformer, categories),
            ]
        )

        return preprocessor

    def fit(self, jobs: list[JobData], random_state: np.random.RandomState | None, **kwargs):
        X, categories, numerics = jobs_to_pandas(jobs)

        Y = X[["runtime", "memory"]]

        # Drop the runtime and memory columns
        X = X.drop(columns=["runtime", "memory"])

        # Split test and train data
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)

        self.last_fit_dsize = len(X_train)

        self.runtime_model = self._make_model(categories, numerics)
        self.memory_model = self._make_model(categories, numerics)

        self.runtime_model.fit(X_train, y_train["runtime"])
        self.memory_model.fit(X_train, y_train["memory"])

        # Evaluate the model on test
        Y_pred_runtime = self.runtime_model.predict(X_test)
        Y_pred_memory = self.memory_model.predict(X_test)

        self.model_metrics = {
            "runtime": {
                "mpe": self.mean_percent_error(y_test["runtime"], Y_pred_runtime),
                "mse": mean_squared_error(y_test["runtime"], Y_pred_runtime),
            },
            "memory": {
                "mpe": self.mean_percent_error(y_test["memory"], Y_pred_memory),
                "mse": mean_squared_error(y_test["memory"], Y_pred_memory),
            },
        }
        # TODO: Warning if model metrics are larger than a threshold.

    def _resolve(self, model, X, default, resource: str, job_name: str, mpe: float) -> tuple[float, list[str]]:
        """Choose between a model's prediction and the caller's default for one resource."""
        predicted = model.predict(X)[0]

        if predicted <= 0:
            return default, [
                f"Predicted {resource} for job {job_name} is negative.",
                f"Returning default {resource} value.",
            ]

        if predicted >= MAX_PREDICTION_FACTOR * default:
            return default, [
                (
                    f"Predicted {resource} for job {job_name} is more than "
                    f"{MAX_PREDICTION_FACTOR} times larger than default."
                ),
                f"Returning default {resource} value.",
            ]

        if mpe > MPE_THRESHOLD:
            return predicted, [
                (
                    f"{resource.capitalize()} prediction for job {job_name} is not within "
                    f"{MPE_THRESHOLD}% of actual value."
                ),
            ]

        return predicted, []

    def predict(self, job: JobData) -> tuple[JobData, list[str]]:
        if self.last_fit_dsize < 10:
            return (
                job,
                "Not enough fitting data points in the fits. Returning default values.",
            )

        X, _, _ = jobs_to_pandas([job])
        warnmsg = []

        job.runtime, runtime_warnings = self._resolve(
            self.runtime_model, X, job.runtime, "runtime", job.job_name, self.model_metrics["runtime"]["mpe"]
        )
        warnmsg += runtime_warnings

        job.memory, memory_warnings = self._resolve(
            self.memory_model, X, job.memory, "memory", job.job_name, self.model_metrics["memory"]["mpe"]
        )
        warnmsg += memory_warnings

        return job, warnmsg
