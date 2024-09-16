import os
import pytest
import h5py
import numpy as np
from slurmise.job_database import JobDatabase

from .utils import print_hdf5


@pytest.fixture
def empty_h5py_file(tmp_path):
    d = tmp_path
    p = d / "slurmise.h5"
    return p


def test_close(empty_h5py_file):
    with JobDatabase.get_database(empty_h5py_file):
        pass


def test_rqd(empty_h5py_file):
    with JobDatabase.get_database(empty_h5py_file) as db:
        db.record(
            job_name="test_job", variables={"runtime": 5, "memory": 100}, slurm_id="1"
        )
        # assert commit_value == 1

        db.record(
            job_name="test_job",
            variables={"runtime": 6, "memory": 128, "filesizes": [123, 512, 128]},
            slurm_id="2",
        )

        # print_hdf5(db.db)

        excepted_results = [
            {"runtime": 5, "memory": 100},
            {"runtime": 6, "memory": 128, "filesizes": np.array([123, 512, 128])},
        ]

        query_result = db.query(job_name="test_job")

        np.testing.assert_equal(query_result, excepted_results)
        db.record(
            job_name="test_job2",
            variables={"runtime": 6, "memory": 128, "filesizes": [123, 512, 128]},
            slurm_id="2",
        )
        db.delete(job_name="test_job")
        query_result = db.query(job_name="test_job")
        assert query_result == []
        query_result = db.query(job_name="test_job2")
        excepted_results = [
            {
                "runtime": 6,
                "memory": 128,
                "filesizes": np.array([123, 512, 128]),
            }
        ]
        np.testing.assert_equal(query_result, excepted_results)
