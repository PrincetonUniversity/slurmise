from slurmise.__main__ import main
from slurmise import job_database
from .utils import print_hdf5
import pytest
from click.testing import CliRunner
from .test_job_database import empty_h5py_file

def test_main():
    pass

def test_raw_record(empty_h5py_file):
    """Test the raw_record command."""
    runner = CliRunner()
    result = runner.invoke(main, ["--database", empty_h5py_file, "raw-record", "--job-name", "test", "--slurm-id", "1234", "--numerical", '"n":3,"q":17.4', "--categorical", '"a":1,"b":2', "--cmd", "sleep 2"])
    assert result.exit_code == 0

    with job_database.JobDatabase.get_database(empty_h5py_file) as db:
        print_hdf5(db.db)

        assert db[0].job_name == "test"
        assert db[0].slurm_id == "1234"
        assert db[0].numerical == {"n": 3, "q": 17.4}
        assert db[0].categorical == {"a": 1, "b": 2}
        assert db[0].cmd == "sleep 2"