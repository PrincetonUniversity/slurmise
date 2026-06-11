import pytest

from slurmise import slurm


def test_parse_slurm_job_metadata(sacct_mock):
    sacct_mock()

    expected_metadata = {
        "CPUs": 96,
        "elapsed_seconds": 97201,
        "job_name": "finetune_vicuna_7b",
        "max_rss": 70917,
        "memory_per_cpu": {
            "infinite": False,
            "number": 0,
            "set": False,
        },
        "memory_per_node": {
            "infinite": False,
            "number": 729088,
            "set": True,
        },
        "partition": "pli-c",
        "slurm_id": 58976578,
        "state": "RUNNING",
        "step_id": "extern",
    }

    assert slurm.parse_slurm_job_metadata("58976578") == expected_metadata
    assert slurm.parse_slurm_job_metadata("58976578", step_id="extern") == expected_metadata


def test_parse_slurm_job_metadata2(sacct_mock):
    sacct_mock(task_count=5)

    expected_metadata = {
        "CPUs": 96,
        "elapsed_seconds": 97201,
        "job_name": "finetune_vicuna_7b",
        "max_rss": 118195,
        "memory_per_cpu": {
            "infinite": False,
            "number": 0,
            "set": False,
        },
        "memory_per_node": {
            "infinite": False,
            "number": 729088,
            "set": True,
        },
        "partition": "pli-c",
        "slurm_id": 58976578,
        "state": "RUNNING",
        "step_id": "extern",
    }

    assert slurm.parse_slurm_job_metadata("58976578") == expected_metadata
    assert slurm.parse_slurm_job_metadata("58976578", step_id="extern") == expected_metadata


def test_parse_slurm_job_metadata_env_fallback(sacct_mock, monkeypatch, no_slurm_env):
    calls = sacct_mock()
    monkeypatch.setenv("SLURM_JOB_ID", "58976578")

    metadata = slurm.parse_slurm_job_metadata()

    assert calls == ["58976578"]
    assert metadata["slurm_id"] == 58976578


def test_parse_slurm_job_metadata_no_id_no_env(sacct_mock, no_slurm_env):
    sacct_mock()

    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurm.parse_slurm_job_metadata()


def test_get_current_job_id(monkeypatch, no_slurm_env):
    assert slurm.get_current_job_id() is None

    monkeypatch.setenv("SLURM_JOBID", "111")
    assert slurm.get_current_job_id() == "111"

    # the modern variable name takes precedence
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    assert slurm.get_current_job_id() == "222"


def test_resolve_job_id_explicit(no_slurm_env):
    assert slurm.resolve_job_id("1234") == "1234"
    assert slurm.resolve_job_id(1234) == "1234"
    assert slurm.resolve_job_id("1234", step_id="0") == "1234.0"
    assert slurm.resolve_job_id("1234", step_id="extern") == "1234.extern"
    # an already combined id is not double-appended
    assert slurm.resolve_job_id("1234.0", step_id="0") == "1234.0"


def test_resolve_job_id_env_fallback(monkeypatch, no_slurm_env):
    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    assert slurm.resolve_job_id() == "1234"
    # the core issue-79 regression: step without an explicit job id
    assert slurm.resolve_job_id(step_id="0") == "1234.0"


def test_resolve_job_id_no_id_no_env(no_slurm_env):
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurm.resolve_job_id()
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurm.resolve_job_id(step_id="0")


def test_split_job_id():
    assert slurm.split_job_id("1234") == ("1234", None)
    assert slurm.split_job_id("1234.0") == ("1234", "0")
    assert slurm.split_job_id("1234.extern") == ("1234", "extern")
    # array job ids have no dot and pass through
    assert slurm.split_job_id("123_4") == ("123_4", None)
    assert slurm.split_job_id("123_4.0") == ("123_4", "0")


def test_get_slurm_job_sacct_outside_slurm_job(monkeypatch, no_slurm_env):
    """Post-mortem recording with an explicit id works outside of a SLURM job."""
    monkeypatch.setattr(
        "slurmise.slurm.subprocess.check_output",
        lambda cmd: b'{"jobs": []}',  # noqa: ARG005
    )

    assert slurm.get_slurm_job_sacct("1234") == {"jobs": []}
