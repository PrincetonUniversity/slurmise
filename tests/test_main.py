import numpy as np
import pytest
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


def test_record(simple_toml, sacct_mock):
    sacct_mock(task_count=1, mem_count=232 * 2**20)  # parses to max_rss=232, elapsed=97201

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
                slurm_id="1234",
                runtime=1620,
                memory=232,
                categories={"complexity": "simple"},
                numerics={"threads": 2},
                cmd=None,
            ),
        ]

        query = JobData(
            job_name="nupack",
            categories={"complexity": "simple"},
        )
        query_result = db.query(query)

        assert query_result == excepted_results


def test_record_duplicate_slurm_id_fails(simple_toml, sacct_mock):
    # Job 1
    sacct_mock(task_count=1, mem_count=232 * 2**20)  # parses to max_rss=232, elapsed=97201

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

    # Job 2
    sacct_mock(task_count=1, mem_count=132 * 2**20)  # Now max_rss=132

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
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "already exists" in str(result.exception)

    # test that the recorded job was the first one and it wasn't overwritten
    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        excepted_results = [
            JobData(
                job_name="nupack",
                slurm_id="1234",
                runtime=1620,
                memory=232,  # This indicates it's the first job
                categories={"complexity": "simple"},
                numerics={"threads": 2},
                cmd=None,
            ),
        ]

        query = JobData(
            job_name="nupack",
            categories={"complexity": "simple"},
        )
        query_result = db.query(query)

        assert query_result == excepted_results


def test_record_duplicate_slurm_id_with_step_id_fails(simple_toml, sacct_mock):
    """Recording the same slurm-id + step-id combination twice should fail the same
    way as a plain duplicate slurm-id, since the resolved slurm_id ("1234.0") is identical."""
    # Job 1
    sacct_mock(step_name="0", task_count=1, mem_count=232 * 2**20)  # parses to max_rss=232, elapsed=97201

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "record",
            "--slurm-id",
            "1234",
            "--step-id",
            "0",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0

    # Job 2: same slurm-id and same step-id
    sacct_mock(step_name="0", task_count=1, mem_count=132 * 2**20)  # Now max_rss=132

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "record",
            "--slurm-id",
            "1234",
            "--step-id",
            "0",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "already exists" in str(result.exception)

    # test that the recorded job was the first one and it wasn't overwritten
    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        excepted_results = [
            JobData(
                job_name="nupack",
                slurm_id="1234.0",
                runtime=1620,
                memory=232,  # This indicates it's the first job
                categories={"complexity": "simple"},
                numerics={"threads": 2},
                cmd=None,
            ),
        ]

        query = JobData(
            job_name="nupack",
            categories={"complexity": "simple"},
        )
        query_result = db.query(query)

        assert query_result == excepted_results


@pytest.mark.xfail(
    reason="Behavior not yet decided (see #84): recording a job with slurm_id=1234, step_id=0 "
    "is recoreded as '1234.0'. If we then try to record a job with the same slurm_id but "
    "with no --step-id, we're currently recorded a separate job '1234'. This is unlikely to happen often "
    "but one option would be to try and auto-increment step-id and have the second job be '1234.1'",
    strict=True,
)
def test_record_missing_step_id_after_stepped_job_should_increment_step_id(simple_toml, sacct_mock):
    # Job 1: recorded with an explicit step-id of 0
    sacct_mock(step_name="0", task_count=1, mem_count=232 * 2**20)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "record",
            "--slurm-id",
            "1234",
            "--step-id",
            "0",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0

    # Job 2: same base slurm-id, but no step-id given.
    # Desired future behavior: this should be recognized as another step of job 1
    # and auto-assigned slurm_id "1234.1".
    sacct_mock(task_count=1, mem_count=132 * 2**20)

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

    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        query = JobData(
            job_name="nupack",
            categories={"complexity": "simple"},
        )
        query_result = db.query(query)

        recorded_slurm_ids = sorted(job.slurm_id for job in query_result)

    assert recorded_slurm_ids == ["1234.0", "1234.1"]


def test_raw_record(simple_toml, sacct_mock):
    """Test the raw_record command."""
    sacct_mock(task_count=1, mem_count=232 * 2**20)

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
            "--numerics",
            '"n":3,"q":17.4',
            "--categories",
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
                categories={"a": 1, "b": 2},
                numerics={"n": 3, "q": 17.4},
                memory=232,
                runtime=1620,
                cmd=None,
            ),
        ]

        query = JobData(
            job_name="test",
            categories={"a": 1, "b": 2},
        )
        query_result = db.query(query)

        assert query_result == excepted_results

    # test the db can print the new values
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "print",
        ],
    )
    assert result.exit_code == 0

    split_std = result.stdout.split("\n")
    assert split_std[0] == "test"
    assert split_std[1].split("-")[-1] == " a=1"
    assert split_std[2].split("-")[-1] == " b=2"
    assert split_std[3].split("-")[-1] == " 1234"


def test_raw_record_with_usage_skips_sacct(simple_toml, no_sacct):
    """--memory and --runtime are taken at face value, with no sacct call."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "raw-record",
            "--job-name",
            "nupack",
            "--slurm-id",
            "1234",
            "--numerics",
            '"threads":2',
            "--categories",
            '"complexity":"simple"',
            "--used-mbs",
            "512",
            "--used-seconds",
            "60",
        ],
    )
    assert result.exit_code == 0

    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        query = JobData(job_name="nupack", categories={"complexity": "simple"})
        assert db.query(query) == [
            JobData(
                job_name="nupack",
                slurm_id="1234",
                runtime=60,
                memory=512,
                categories={"complexity": "simple"},
                numerics={"threads": 2},
                cmd=None,
            ),
        ]


@pytest.mark.parametrize(
    "usage_flags",
    [["--used-mbs", "512"], ["--used-seconds", "60"]],
    ids=["mbs-only", "seconds-only"],
)
def test_raw_record_partial_usage_is_rejected(simple_toml, usage_flags, no_sacct):
    """Half the usage is refused rather than silently dropped and sacct is not consulted."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "raw-record",
            "--job-name",
            "nupack",
            "--slurm-id",
            "1234",
            "--numerics",
            '"threads":2',
            "--categories",
            '"complexity":"simple"',
            *usage_flags,
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "BOTH used memory and runtime are required" in str(result.exception)

    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        assert db.query(JobData(job_name="nupack", categories={"complexity": "simple"})) == []


def test_print_explicit_h5_path(tmp_path):
    """print accepts a bare .h5 path with no toml configuration."""
    h5 = tmp_path / "mydb.h5"

    with job_database.JobDatabase.get_database(h5) as db:
        db.record(JobData(job_name="test_job", slurm_id="1", runtime=5, memory=100))

    runner = CliRunner()
    result = runner.invoke(main, ["print", str(h5)])

    assert result.exit_code == 0
    assert "test_job" in result.stdout


def test_print_h5_path_overrides_toml(tmp_path, simple_toml):
    """An explicit h5 path takes precedence over the --toml database."""
    h5 = tmp_path / "explicit.h5"

    with job_database.JobDatabase.get_database(h5) as db:
        db.record(JobData(job_name="explicit_job", slurm_id="1", runtime=5, memory=100))

    runner = CliRunner()
    result = runner.invoke(main, ["--toml", simple_toml.toml, "print", str(h5)])

    assert result.exit_code == 0
    assert "explicit_job" in result.stdout


def test_print_missing_default(tmp_path, monkeypatch):
    """print errors clearly when no database can be resolved."""
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(main, ["print"])

    assert result.exit_code == 1
    assert "No database found" in result.output


def test_record_step_id_without_slurm_id(simple_toml, monkeypatch, no_slurm_env, sacct_mock):
    """Regression test for issue 79: `record --step-id N` without `--slurm-id` inside
    a SLURM job must resolve the job id from the environment."""
    sacct_calls = sacct_mock(step_name="0", task_count=1, mem_count=232 * 2**20)
    monkeypatch.setenv("SLURM_JOB_ID", "4321")

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "record",
            "--step-id",
            "0",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0
    assert sacct_calls == ["4321"]

    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        results = db.query(JobData(job_name="nupack", categories={"complexity": "simple"}))
        assert [job.slurm_id for job in results] == ["4321.0"]


def test_record_step_id_outside_slurm_job(simple_toml, no_slurm_env):
    """Without --slurm-id and outside a SLURM job, record fails with a clear error."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "record",
            "--step-id",
            "0",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "SLURM_JOB_ID" in str(result.exception)


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
    if result.exception:  # pragma: no cover
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
    assert predicted_runtime[0] == "Predicted runtime"
    np.testing.assert_allclose(float(predicted_runtime[1]), 9.29, rtol=0.01)
    assert predicted_memory[0] == "Predicted memory"
    np.testing.assert_allclose(float(predicted_memory[1]), 10168.72, rtol=0.01)

    result = runner.invoke(
        main,
        [
            "--toml",
            nupack_toml.toml,
            "raw-predict",
            "--job-name=nupack",
            '--numerics="cpus":3,"sequences":6543',
            "--cmd='nupack monomer -c 3 -S 6543'",
        ],
    )

    assert result.exit_code == 0
    tmp_stdout = result.stdout.split("\n")
    predicted_runtime = tmp_stdout[0].split(":")
    predicted_memory = tmp_stdout[1].split(":")
    assert predicted_runtime[0] == "Predicted runtime"
    np.testing.assert_allclose(float(predicted_runtime[1]), 9.29, rtol=0.01)
    assert predicted_memory[0] == "Predicted memory"
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
    assert predicted_runtime[0] == "Predicted runtime"
    assert float(predicted_runtime[1]) == 60
    assert predicted_memory[0] == "Predicted memory"
    assert float(predicted_memory[1]) == 1000
    assert "Warnings:" in result.stderr

    # Test that slurmise returns the default values when the predicted values are not possible.
    result = runner.invoke(
        main,
        [
            "--toml",
            nupack_toml.toml,
            "raw-predict",
            "--job-name=nupack",
            '--numerics="cpus":987654,"sequences":4985',
            "--cmd='nupack monomer -c 987654 -S 4985'",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0
    tmp_stdout = result.stdout.split("\n")
    predicted_runtime = tmp_stdout[0].split(":")
    predicted_memory = tmp_stdout[1].split(":")
    assert predicted_runtime[0] == "Predicted runtime"
    assert float(predicted_runtime[1]) == 60
    assert predicted_memory[0] == "Predicted memory"
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
    assert predicted_runtime[0] == "Predicted runtime"
    np.testing.assert_allclose(float(predicted_runtime[1]), 9.29, rtol=0.01)
    assert predicted_memory[0] == "Predicted memory"
    np.testing.assert_allclose(float(predicted_memory[1]), 10168.72, rtol=0.01)


def test_predict_nomodel(nupackdefaults_toml):
    """Test the predict commands of slurmise with no model.
    Running predict before updating (creating) a model will cause the job
    default values to be returned.
    """
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--toml",
            nupackdefaults_toml.toml,
            "predict",
            "nupack monomer -c 987654 -S 4985",
        ],
        catch_exceptions=True,
    )
    assert result.exit_code == 0
    tmp_stdout = result.stdout.split("\n")
    predicted_runtime = tmp_stdout[0].split(":")
    predicted_memory = tmp_stdout[1].split(":")
    assert predicted_runtime[0] == "Predicted runtime"
    assert float(predicted_runtime[1]) == 80
    assert predicted_memory[0] == "Predicted memory"
    assert float(predicted_memory[1]) == 3000
    assert "Warnings:" in result.stderr


def test_parse(simple_toml):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--toml",
            simple_toml.toml,
            "parse",
            "nupack monomer -T 2 -C simple",
        ],
    )
    assert result.exit_code == 0

    assert result.stdout.startswith("Able to parse")
