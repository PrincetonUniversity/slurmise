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


def test_record_and_query(empty_json_file):
    with JobDatabase.get_database(empty_json_file) as db:
        commit_value = db.record(
            job_name="test_job", variables={"runtime": 5, "memory": 100}
        )
        assert commit_value == 1

        commit_value = db.record(
            job_name="test_job",
            variables={"runtime": 6, "memory": 128, "filesizes": [123, 512, 128]},
        )
        assert commit_value == 2

        excepted_results = [
            {"runtime": 5, "memory": 100},
            {"runtime": 6, "memory": 128, "filesizes": [123, 512, 128]},
        ]

        query_result = db.query(job_name="test_job")
        assert query_result == excepted_results
