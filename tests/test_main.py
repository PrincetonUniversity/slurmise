import pytest
from click.testing import CliRunner

from slurmise import job_database
from slurmise.__main__ import main
from slurmise.job_data import JobData


@pytest.fixture
def empty_h5py_file(tmp_path):
    d = tmp_path
    p = d / "slurmise.h5"
    return p


@pytest.fixture
def simple_toml(tmp_path):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
    '''
    [slurmise]
    base_dir = "slurmise_dir"

    [slurmise.job.nupack]
    job_spec = "monomer -T {threads:numeric} -C {complexity:category}"
    ''')
    return p


def test_record(empty_h5py_file, simple_toml, monkeypatch):
    mock_metadata = {
        "slurm_id": "1234",
        "job_name": "nupack",
        "state": "COMPLETED",
        "partition": "",
        "elapsed_seconds": 97201,
        "CPUs": 1,
        "memory_per_cpu": 0,
        "memory_per_node": 0,
        "max_rss": 232,
    }
    monkeypatch.setattr("slurmise.slurm.parse_slurm_job_metadata", lambda *args,
                        **kwargs: mock_metadata)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--database",
            empty_h5py_file,
            "--toml",
            simple_toml,
            "record",
            "--slurm-id",
            "1234",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0

    # test the job was successfully added
    with job_database.JobDatabase.get_database(empty_h5py_file) as db:
        excepted_results = [
            JobData(
                job_name="nupack",
                slurm_id="1234",
                runtime=97201,
                memory=232,
                categorical={"complexity": 'simple'},
                numerical={"threads": 2},
                cmd=None,
            ),
        ]

        query = JobData(
            job_name="nupack",
            categorical={"complexity": "simple"},
        )
        query_result = db.query(query)

        assert query_result == excepted_results


def test_raw_record(empty_h5py_file):
    """Test the raw_record command."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--database",
            empty_h5py_file,
            "raw-record",
            "--job-name",
            "test",
            "--slurm-id",
            "1234",
            "--numerical",
            '"n":3,"q":17.4',
            "--categorical",
            '"a":1,"b":2',
            "--cmd",
            "sleep 2",
        ],
    )
    assert result.exit_code == 0

    # test the job was successfully added
    with job_database.JobDatabase.get_database(empty_h5py_file) as db:
        excepted_results = [
            JobData(
                job_name="test",
                slurm_id="1234",
                categorical={"a": 1, "b": 2},
                numerical={"n": 3, "q": 17.4},
                cmd=None,
            ),
        ]

        query = JobData(
            job_name="test",
            categorical={"a": 1, "b": 2},
        )
        query_result = db.query(query)

        assert query_result == excepted_results
