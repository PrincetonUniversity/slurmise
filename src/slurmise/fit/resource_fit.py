import pathlib
import json
import hashlib

import pandas as pd
import numpy as np

from typing import Optional
from dataclasses import dataclass, asdict, field
from ..job_data import JobData
from datetime import datetime


BASEMODELPATH = pathlib.Path.home() / ".slurmise/models/"


@dataclass(kw_only=True)
class ResourceFit:
    query: JobData
    last_fit_dsize: int = 0
    fit_timestamp: datetime = field(default_factory=datetime.now)
    model_metrics: dict = field(default_factory=dict)
    path: Optional[pathlib.Path] = None

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
            'class': cls.__name__,
            "job_name": query.job_name,
            **query.categorical,
        }
        hash_info_tuple = tuple(hash_info.items())

        # Get and MD5 hash of information
        hash_val = hashlib.md5(str(hash_info_tuple).encode('utf-8')).hexdigest()

        return hash_val

    @classmethod
    def _make_model_path(cls, query) -> pathlib.Path:
        """
        This method returns the path to the model's directory.

        The model's path is a function of the model's type and the hash of its query.
        """
        hash_val = cls._get_model_info_hash(query)
        return pathlib.Path(BASEMODELPATH) / cls.__name__ / hash_val

    def save(self, model_params: dict = {}):
        """This method saves the basic information of the model, such as its query,
        when it was last fit, the dataset size of the latest fit, and the type of
        the model.
        """
        self.path.mkdir(parents=True, exist_ok=True)    
        with open(str(self.path / "fits.json"), "w") as save_file:
            info = asdict(self)

            # Convert path to string
            info['path'] = str(info['path'])

            # Convert datetime to string
            info['fit_timestamp'] = info['fit_timestamp'].isoformat()

            info.update(model_params)
            json.dump(info, save_file)

    @classmethod
    def load(cls, query: JobData | None = None, path: str | None = None) -> "ResourceFit":
        """
        This method loads a model from a file. The model is loaded from the path
        provided, or from the path generated from the query.

        :param query: The query used to generate the model
        :type query: JobData
        :param path: The path to the model
        :type path: str
        :return: The model
        :rtype: ResourceFit

        """
        if path is None and query is None:
            raise ValueError("Either query or path must be provided")

        if path is None:
            path = cls._make_model_path(query)

        if isinstance(path, str):
            path = pathlib.Path(path)

        with open(str(path / "fits.json")) as load_file:
            info = json.load(load_file)

        # Convert datetime from isoformat string to datetime object
        info['fit_timestamp'] = datetime.fromisoformat(info['fit_timestamp'])

        return cls(**info)

    def predict(self, job: JobData) -> tuple[float, float]:
        raise NotImplementedError

    def fit(self, jobs: list[JobData], random_state: np.random.RandomState | None, **kwargs):
        raise NotImplementedError

    @staticmethod
    def jobs_to_pandas(jobs: list[JobData]):
        """
        Convert a list of JobData objects to a pandas DataFrame. The DataFrame will have
        columns for each categorical and numerical feature, and will not include the job_name,
        slurm_id, memory, or runtime columns.

        :param jobs: A list of JobData objects
        :type jobs: list[JobData]
        :return: A pandas DataFrame with columns for each categorical and numerical feature
        :rtype: pd.DataFrame

        """

        df = pd.json_normalize([asdict(job) for job in jobs])

        # Convert categorical columns to category type
        for col in df.columns:
            if col.startswith("categorical."):
                df[col] = df[col].astype("category")

        # Rename the categorical columns, drop .categorical prefix
        df.columns = [col.replace("categorical.", "") for col in df.columns]

        # Do the same for numerical columns
        for col in df.columns:
            new_col_name = col.replace("numerical.", "")

            if col.startswith("numerical."):

                if df[col].dtype == "object":

                    # Check if they are all numpy arrays
                    if all([isinstance(l, np.ndarray) for l in df[col]]):
                        # Check if the column is a numpy array of all the same size
                        sizes = set([l.shape for l in df[col]])

                        if len(sizes) == 1:
                            # If all the same size, expand each element of the numpy array into a new column
                            col_df = pd.DataFrame(np.vstack([s.flatten() for s in df.loc[0:10, 'numerical.sequences']]))
                            col_df.columns = [f"{new_col_name}_{i}" for i in range(col_df.shape[1])]

                            # Drop the original column and add the new columns
                            df = df.drop(columns=[col])
                            df = pd.concat([df, col_df], axis=1)

                        else:
                            raise ValueError(f"Numerical feature {new_col_name} is an a numpy array of different sizes. "
                                             f"Numpy arrays are supported only if they are all the same size.")

                    else:
                        raise ValueError("Numerical columns must be scalars or equal length numpy arrays")

        df.columns = [col.replace("numerical.", "") for col in df.columns]

        # Get the numerical columns
        df = df.drop(columns=["job_name", "slurm_id", "cmd"])

        # Transform features
        categorical_features = [
            name for name in df.columns if df[name].dtype == "category"
        ]
        numerical_features = [
            name for name in df.columns if name not in categorical_features and name not in ["memory", "runtime"]
        ]

        return df, categorical_features, numerical_features
