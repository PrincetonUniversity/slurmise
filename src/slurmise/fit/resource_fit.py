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
    def load(self) -> "ResourceFit":
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
