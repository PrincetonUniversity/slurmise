import pathlib
import json
from typing import Optional
from dataclasses import dataclass, asdict
from ..job_data import JobData
from datetime import datetime

BASEMODELPATH = "~/.slurmise/models/"


@dataclass
class ResourceFit:
    query: JobData
    model_type: str
    last_fit_dsize: int
    fit_timestamp: datetime
    path: Optional[str] = None

    def save(self):
        """This method saves the basic information of the model, such as its query,
        when it was last fit, the dataset size of the latest fit, and the type of
        the model.
        """
        hash_info = {"job_name": self.query.job_name}
        hash_info.update(self.query.categorical)
        hash_val = hash(hash_info)
        self.path = (
            pathlib.Path(BASEMODELPATH) / self.model_type / hash_val / "fit.json"
        )
        with open(str(self.path), "w") as save_file:
            info = asdict(self)
            json.dump(save_file, info)

    @staticmethod
    def load(self, query: JobData) -> "ResourceFit":
        """This method loads a ResourceFit.

        :param query: The query of the model
        :type query: JobData
        :return: Returns the basic information of the fit.
        :rtype: ResourceFit
        """
        if self.path is None:
            hash_info = {"job_name": self.query.job_name}
            hash_info.update(self.query.categorical)
            hash_val = hash(hash_info)
            self.path = (
                pathlib.Path(BASEMODELPATH) / self.model_type / hash_val / "fit.json"
            )
        with open(str(self.path)) as load_file:
            info = json.load(load_file)

        return ResourceFit(**info)

    def predict(self, job: JobData):
        raise NotImplementedError

    def fit(self, jobs: list[JobData], **kargs):
        raise NotImplementedError
