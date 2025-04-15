import pytest
import shutil
import numpy as np
from pathlib import Path
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
        f"""
    [slurmise]
    base_dir = "{d/'slurmise_dir'}"

    [slurmise.job.nupack]
    job_spec = "monomer -T {{threads:numeric}} -C {{complexity:category}}"
    """
    )
    return p, d / "slurmise_dir" / "slurmise.h5"


@pytest.fixture
def simple_toml2(tmp_path):
    d = tmp_path
    p = d / "slurmise.toml"
    p.write_text(
        f"""
    [slurmise]
    base_dir = "{d/'slurmise_dir'}"
    db_filename = "nupack2.h5"

    [slurmise.job.nupack]
    job_spec = "monomer -c {{cpus:numeric}} -S {{sequences:numeric}}"
    """
    )
    return p, d / "slurmise_dir" / "nupack2.h5"


def test_record(simple_toml, monkeypatch):
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
    monkeypatch.setattr(
        "slurmise.slurm.parse_slurm_job_metadata", lambda *args, **kwargs: mock_metadata
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml[0],
            "record",
            "--slurm-id",
            "1234",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0
    # test the job was successfully added
    with job_database.JobDatabase.get_database(simple_toml[1]) as db:
        excepted_results = [
            JobData(
                job_name="nupack",
                slurm_id="1234",
                runtime=97201,
                memory=232,
                categorical={"complexity": "simple"},
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


def test_raw_record(simple_toml):
    """Test the raw_record command."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml[0],
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
    with job_database.JobDatabase.get_database(simple_toml[1]) as db:
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


def test_update_predict(simple_toml2):
    """Test the update and predict commands of slurmise.
    Initially, we run the update command to get the models for the nupack job.
    After the models are created, we run the predict command to predict the runtime and memory of a job.
    Two tests are run. The first predicts a runtime and memory values for a job that
    makes sense. The second test returns a runtime and memory values that are not
    possible. Because we cannot know the exact numbers we check of the expected strings.
    """
    Path.mkdir(simple_toml2[1].parent, exist_ok=True, parents=True)
    shutil.copyfile(
        "./tests/nupack2.h5",
        simple_toml2[1],
    )
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml2[0],
            "update-model",
            "nupack monomer -c 1 -S 4985",
        ],
        catch_exceptions=True,
    )
    if result.exception:
        print(f"Exception: {result.exception}")
    assert result.exit_code == 0

    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml2[0],
            "predict",
            "nupack monomer -c 3 -S 6543",
        ],
    )
    assert result.exit_code == 0
    tmp_stdout = result.stdout.split("\n")
    predicted_runtime = tmp_stdout[0].split(":")
    predicted_memory = tmp_stdout[1].split(":")
    assert "Predicted runtime" == predicted_runtime[0]
    np.testing.assert_allclose(float(predicted_runtime[1]), 9.29, rtol=0.01)
    assert "Predicted memory" == predicted_memory[0]
    np.testing.assert_allclose(float(predicted_memory[1]), 10168.72, rtol=0.01)

    # Test that slurmise returns the default values when the predicted values are not possible.
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml2[0],
            "predict",
            "nupack monomer -c 987654 -S 4985",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0
    tmp_stdout = result.stdout.split("\n")
    predicted_runtime = tmp_stdout[0].split(":")
    predicted_memory = tmp_stdout[1].split(":")
    assert "Predicted runtime" == predicted_runtime[0]
    assert float(predicted_runtime[1]) == 60
    assert "Predicted memory" == predicted_memory[0]
    assert float(predicted_memory[1]) == 1000
    assert "Warnings:" in result.stderr
