import multiprocessing
import time
from pathlib import Path
from unittest import mock

import pytest

from slurmise import job_database
from slurmise.api import Slurmise
from slurmise.job_data import JobData
from tests.conftest import TomlReturn


def slurmise_record(toml, process_id, error_queue):
    def mock_metadata(kwargs):
        return {
            "slurm_id": kwargs["slurm_id"],
            "job_name": "nupack",
            "state": "COMPLETED",
            "partition": "",
            "elapsed_seconds": 97201,
            "CPUs": 1,
            "memory_per_cpu": 0,
            "memory_per_node": 0,
            "max_rss": 232,
            "step_id": "external",
        }

    try:
        time.sleep(process_id * 0.1)
        with mock.patch(
            "slurmise.slurm.parse_slurm_job_metadata",
            side_effect=lambda *args, **kwargs: mock_metadata(kwargs),
        ):
            slurmise = Slurmise(toml)
            time.sleep(process_id * 0.1)
            for i in range(10):
                slurmise.record("nupack monomer -T 2 -C simple", slurm_id=str(process_id * 100 + i))
                time.sleep(process_id * 0.1)
    except Exception as e:  # noqa: BLE001
        error_queue.put(f"PID {process_id}: {e}")


def test_multiple_slurmise_instances(simple_toml):
    processes = []
    error_queue = multiprocessing.Queue()
    for i in range(10):
        p = multiprocessing.Process(target=slurmise_record, args=(simple_toml.toml, i, error_queue))
        processes.append(p)
        p.start()

    [p.join() for p in processes]

    if not error_queue.empty():
        while not error_queue.empty():
            print(error_queue.get())
        pytest.fail("Child prcess had error")


def test_job_data_from_dict(simple_toml):
    slurmise = Slurmise(simple_toml.toml)
    result = slurmise.job_data_from_dict(
        {"threads": 3, "complexity": "simple"},
        "nupack",
    )
    assert result.categories == {"complexity": "simple"}
    assert result.numerics == {"threads": 3}


@pytest.mark.parametrize(
    "toml_fixture",
    ["simple_toml", "nupackdefaults_toml", "small_db_toml"],
)
def test_update_all_models(toml_fixture, request):
    toml = request.getfixturevalue(toml_fixture)
    slurmise = Slurmise(toml.toml)
    try:
        slurmise.update_all_models()
    except ValueError as e:
        # If there is not enough data to fit a model, a ValueError is raised
        # by sklearn train_test_split. Currently happening with small_db_toml fixture
        # because there is only one job with "filesizes" numeric feature.
        if str(e).startswith("Cannot have number of splits n_splits="):
            pass


# (mode, complexity, record count, runtime/memory slope) per category combination.
CATEGORIES = (
    ("fast", "simple", 20, 3),
    ("slow", "complex", 15, 7),
)


@pytest.fixture
def two_categories_toml(tmp_path):
    """A job with two category variables and two category combinations.

    The job spec lists `mode` before `complexity` so the parsed insertion order differs
    from the sorted order the database stores, which is what makes an order dependent
    model hash observable. The two combinations hold different numbers of records so
    that a loaded model's last_fit_dsize identifies which one it was fit on.
    """
    base_dir = tmp_path / "slurmise_dir"
    toml = tmp_path / "slurmise.toml"
    toml.write_text(
        f"""
    [slurmise]
    base_dir = "{base_dir}"
    db_filename = "two_categories.h5"

    [slurmise.job.nupack]
    job_spec = "monomer -c {{cpus}} -M {{mode}} -C {{complexity}}"
    [slurmise.job.nupack.variables]
    cpus = {{type = "numeric"}}
    mode = {{type = "category"}}
    complexity = {{type = "category"}}
    """
    )

    db_path = base_dir / "two_categories.h5"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with job_database.JobDatabase.get_database(str(db_path)) as database:
        for mode, complexity, record_count, slope in CATEGORIES:
            for i in range(record_count):
                cpus = i + 1
                database.record(
                    JobData(
                        job_name="nupack",
                        slurm_id=f"{mode}_{complexity}_{i}",
                        runtime=slope * cpus + 10,
                        memory=100 * slope * cpus + 500,
                        numerics={"cpus": cpus},
                        categories={"mode": mode, "complexity": complexity},
                    )
                )

    return TomlReturn(str(toml), str(db_path))


def test_update_all_models_saves_each_category_separately(two_categories_toml):
    """Each category gets its own model directory instead of clobbering a shared one (#76)."""
    slurmise = Slurmise(two_categories_toml.toml)
    slurmise.update_all_models()

    base_path = Path(slurmise.configuration.slurmise_base_dir)
    assert len(list(base_path.glob("*/*/fits.json"))) == 2

    # A loaded model's fit size identifies the category it was fit on; if the two
    # fits shared a directory the second would have overwritten the first.
    model_class = slurmise.configuration.get_model_class("nupack")
    fit_sizes = {}
    for mode, complexity, record_count, _ in CATEGORIES:
        query = JobData(job_name="nupack", categories={"mode": mode, "complexity": complexity})
        model_path = model_class._make_model_path(query, base_path=base_path)
        fit_sizes[mode] = (model_class.load(query=query, path=model_path).last_fit_dsize, int(record_count * 0.8))

    assert all(actual == expected for actual, expected in fit_sizes.values()), fit_sizes


def test_update_model_without_cmd_then_predict(two_categories_toml):
    """Fitting every category of a job by name leaves each one predictable.

    This is the round trip that catches an order dependent model hash: update writes
    using categories rebuilt from the database, predict reads using categories parsed
    from the command, and the two must resolve to the same directory.
    """
    slurmise = Slurmise(two_categories_toml.toml)
    slurmise.update_model(None, "nupack")

    for mode, complexity, _, slope in CATEGORIES:
        predicted, warnings = slurmise.predict(f"monomer -c 5 -M {mode} -C {complexity}", "nupack")

        assert "Not enough fitting data points in the fits." not in warnings
        assert predicted.runtime == pytest.approx(slope * 5 + 10, rel=0.01)
        assert predicted.memory == pytest.approx(100 * slope * 5 + 500, rel=0.01)


def test_raw_record_uses_env_slurm_id(simple_toml, monkeypatch, no_slurm_env, sacct_mock):
    """When slurm_id is None, raw_record should fall back to the SLURM_JOB_ID env var."""
    sacct_calls = sacct_mock()
    monkeypatch.setenv("SLURM_JOB_ID", "99999")

    job = JobData(job_name="nupack", slurm_id=None)
    slurmise = Slurmise(simple_toml.toml)
    slurmise.raw_record(job)

    assert job.slurm_id == "99999"
    assert sacct_calls == ["99999"]


def test_raw_record_raises_when_no_slurm_id(simple_toml, no_slurm_env):
    """When slurm_id is None and SLURM_JOB_ID is not set, raise a descriptive ValueError."""
    job = JobData(job_name="nupack", slurm_id=None)

    slurmise = Slurmise(simple_toml.toml)
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurmise.raw_record(job)


def test_record_step_id_without_slurm_id(simple_toml, monkeypatch, no_slurm_env, sacct_mock):
    """Regression test for issue 79: --step-id without --slurm-id must resolve the
    job id from the environment instead of producing the literal id "None.<step>"."""
    sacct_calls = sacct_mock(step_name="0", task_count=1, mem_count=232 * 2**20)
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

    slurmise = Slurmise(simple_toml.toml)
    slurmise.record("nupack monomer -T 2 -C simple", step_id="0")

    assert sacct_calls == ["1234"]

    with job_database.JobDatabase.get_database(simple_toml.db) as db:
        results = db.query(JobData(job_name="nupack", categories={"complexity": "simple"}))
        assert [job.slurm_id for job in results] == ["1234.0"]
        assert results[0].memory == 232
        assert results[0].runtime == 1620


def test_record_step_id_without_slurm_id_or_env(simple_toml, no_slurm_env):
    """With a step_id but neither slurm_id nor environment, fail with a clear error."""
    slurmise = Slurmise(simple_toml.toml)
    with pytest.raises(ValueError, match="SLURM_JOB_ID"):
        slurmise.record("nupack monomer -T 2 -C simple", step_id="0")
