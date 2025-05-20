import numpy as np
from click.testing import CliRunner

from slurmise import job_database
from slurmise.__main__ import main
from slurmise.job_data import JobData


def test_missing_toml():
    """Check that excluding a toml file will fail with error message."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["record", "something"],
    )
    assert result.exit_code == 1
    assert "Slurmise requires a toml file" in result.output
    assert "See readme for more information" in result.output


def test_record(simple_toml, monkeypatch):
    mock_metadata = {
        "slurm_id": "1234",
        "step_id": "0",
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
            simple_toml.toml,
            "record",
            "--slurm-id",
            "1234",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0
    # test the job was successfully added
    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        excepted_results = [
            JobData(
                job_name="nupack",
                slurm_id="1234.0",
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
            simple_toml.toml,
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
    with job_database.JobDatabase.get_database(simple_toml.db) as db:
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


def test_update_predict(nupack_toml):
    """Test the update and predict commands of slurmise.
    Initially, we run the update command to get the models for the nupack job.
    After the models are created, we run the predict command to predict the runtime and memory of a job.
    Two tests are run. The first predicts a runtime and memory values for a job that
    makes sense. The second test returns a runtime and memory values that are not
    possible. Because we cannot know the exact numbers we check of the expected strings.
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            nupack_toml.toml,
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
            nupack_toml.toml,
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
            nupack_toml.toml,
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


def test_update_all_predict(nupack_toml):
    """Test the update all and predict commands of slurmise.
    Initially, we run the update command to get the models for the nupack job.
    After the models are created, we run the predict command to predict the runtime and memory of a job.
    Two tests are run. The first predicts a runtime and memory values for a job that
    makes sense. The second test returns a runtime and memory values that are not
    possible. Because we cannot know the exact numbers we check of the expected strings.
    """
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            nupack_toml.toml,
            "update-all",
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
            nupack_toml.toml,
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
