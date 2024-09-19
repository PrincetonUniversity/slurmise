from slurmise.__main__ import main
from slurmise import job_database
from slurmise.job_data import JobData
from .utils import print_hdf5
import pytest
from click.testing import CliRunner
from .test_job_database import empty_h5py_file

def test_main():
    pass

def test_raw_record(empty_h5py_file):
    """Test the raw_record command."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--database", empty_h5py_file, "raw-record", "--job-name", "test",
         "--slurm-id", "1234", "--numerical", '"n":3,"q":17.4', "--categorical",
         '"a":1,"b":2', "--cmd", "sleep 2"])
    assert result.exit_code == 0

    # test the job was successfully added
    with job_database.JobDatabase.get_database(empty_h5py_file) as db:
        excepted_results = [
            JobData(
                job_name="test",
                slurm_id="1234",
                categorical={"a": 1, "b": 2},
                numerical={"n": 3, "q": 17.4},
                cmd=None
            ),
        ]

        query = JobData(
            job_name="test",
            categorical={"a": 1, "b": 2},
        )
        query_result = db.query(query)

        assert query_result == excepted_results
