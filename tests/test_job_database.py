import pytest

from slurmise.job_database import JobDatabase


@pytest.fixture
def empty_json_file(tmp_path):
    d = tmp_path
    p = d / "slurmise.json"
    return p

def test_close(empty_json_file):
    with JobDatabase.get_database(empty_json_file):
        pass
