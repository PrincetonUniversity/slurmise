import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..job_data import JobData

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
            "class": cls.__name__,
            "job_name": query.job_name,
            **query.categorical,
        }
        hash_info_tuple = tuple(hash_info.items())

        # Get and MD5 hash of information
        hash_val = hashlib.md5(str(hash_info_tuple).encode("utf-8")).hexdigest()

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
            # This converts the dataclass to a dictionary. If it is called from a subclass,
            # the subclass's attributes will be included in the dictionary.
            info = asdict(self)

            # Convert path to string
            info["path"] = str(info["path"])

            # Convert datetime to string
            info["fit_timestamp"] = info["fit_timestamp"].isoformat()

            info.update(model_params)
            json.dump(info, save_file)

    @classmethod
    def load(
        cls, query: JobData | None = None, path: str | None = None, **kwargs
    ) -> "ResourceFit":
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
            info["fit_timestamp"] = datetime.fromisoformat(info["fit_timestamp"])
        else:
            info = {
                "query": query,
                "last_fit_dsize": 0,
                "fit_timestamp": datetime.now(),
                "model_metrics": {},
                "path": path,
            }
            info.update(kwargs)

        # Generates an instance of the class from the dictionary. When this is called by
        # a ResourceFit subclass, it includes all attributes of the subclass(es) and the
        # ResourceFit class.

        return cls(**info)

    def predict(self, job: JobData) -> tuple[JobData, list[str]]:
        raise NotImplementedError

    def fit(
        self, jobs: list[JobData], random_state: np.random.RandomState | None, **kwargs
    ):
        raise NotImplementedError
