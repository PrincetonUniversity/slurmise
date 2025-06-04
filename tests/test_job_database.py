import numpy as np
import pytest

from slurmise.job_data import JobData
from slurmise.job_database import JobDatabase


@pytest.fixture
def empty_h5py_file(tmp_path):
    d = tmp_path
    p = d / "slurmise.h5"
    return p


@pytest.fixture
def small_db(empty_h5py_file):
    with JobDatabase.get_database(empty_h5py_file) as db:
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                runtime=6,
                memory=128,
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
                categorical={"option1": "value1", "option2": "value2"},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                numerical={"filesizes": [123, 512, 128]},
                categorical={"option1": "value2"},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="3",
            )
        )
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="4",
                runtime=7,
                memory=100,
                categorical={"option2": "value2", "option1": "value1"},
            )
        )
        yield db


def test_close(empty_h5py_file):
    """Test opening and closing an empty database."""
    with JobDatabase.get_database(empty_h5py_file):
        pass


def test_rqd_flat(empty_h5py_file):
    """Test record, query and delete on jobs without categorical values."""
    with JobDatabase.get_database(empty_h5py_file) as db:
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
            )
        )
        # assert commit_value == 1

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                runtime=6,
                memory=128,
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        excepted_results = [
            JobData(job_name="test_job", slurm_id="1", runtime=5, memory=100),
            JobData(
                job_name="test_job",
                slurm_id="2",
                runtime=6,
                memory=128,
                numerical={"filesizes": np.array([123, 512, 128])},
            ),
        ]

        query_result = db.query(JobData(job_name="test_job"))

        assert query_result == excepted_results

        db.record(
            JobData(
                job_name="test_job2",
                slurm_id="2",
                runtime=6,
                memory=128,
                numerical={"filesizes": [123, 512, 128]},
            )
        )
        db.delete(JobData(job_name="test_job"))
        query_result = db.query(JobData(job_name="test_job"))
        assert query_result == []
        query_result = db.query(JobData(job_name="test_job2"))
        excepted_results = [
            JobData(
                slurm_id="2",
                job_name="test_job2",
                runtime=6,
                memory=128,
                numerical={"filesizes": np.array([123, 512, 128])},
            )
        ]
        np.testing.assert_equal(query_result, excepted_results)


def test_rqd_with_categories(empty_h5py_file):
    with JobDatabase.get_database(empty_h5py_file) as db:
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
                categorical={"option1": "value1", "option2": "value2"},
            )
        )
        # assert commit_value == 1

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2",
                numerical={"filesizes": [123, 512, 128]},
                categorical={"option1": "value2"},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="3",
            )
        )
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="4",
                runtime=7,
                memory=100,
                categorical={"option2": "value2", "option1": "value1"},
            )
        )

        excepted_results = [
            JobData(job_name="test_job", slurm_id="3"),
        ]

        query_result = db.query(JobData(job_name="test_job"))
        assert query_result == excepted_results

        excepted_results = [
            JobData(
                job_name="test_job",
                slurm_id="1",
                runtime=5,
                memory=100,
                categorical={"option1": "value1", "option2": "value2"},
            ),
            JobData(
                job_name="test_job",
                slurm_id="4",
                runtime=7,
                memory=100,
                categorical={"option2": "value2", "option1": "value1"},
            ),
        ]

        query_result = db.query(
            JobData(
                job_name="test_job",
                categorical={"option2": "value2", "option1": "value1"},
            )
        )
        assert query_result == excepted_results

        query_result = db.query(
            JobData(
                job_name="test_job",
                categorical={
                    "option1": "value1",
                    "option2": "value2",
                },
            )
        )
        assert query_result == excepted_results

        excepted_results = [
            JobData(
                job_name="test_job",
                slurm_id="2",
                numerical={"filesizes": np.array([123, 512, 128])},
                categorical={"option1": "value2"},
            )
        ]
        query_result = db.query(
            JobData(
                job_name="test_job",
                categorical={
                    "option1": "value2",
                },
            )
        )
        assert query_result == excepted_results


def test_rqd_with_emptydb(empty_h5py_file):
    """Test record, query and delete on empty database."""
    with JobDatabase.get_database(empty_h5py_file) as db:
        query_result = db.query(
            JobData(
                job_name="test_job",
                categorical={
                    "option1": "value2",
                },
            )
        )
        assert query_result == []

        query_result = db.query(
            JobData(
                job_name="test_job2",
            )
        )
        assert query_result == []

        # no errors
        db.delete(JobData(job_name="test_job3"), delete_all_children=False)
        db.delete(
            JobData(job_name="test_job3", categorical={"option1": "value1"}),
            delete_all_children=False,
        )


def test_iterate_database(small_db):
    """Test access to all jobs and data through an iterator."""
    expected_queries = [
        JobData(job_name='test_job', categorical={'option1': 'value1', 'option2': 'value2'}),
        JobData(job_name='test_job', categorical={'option1': 'value1'}),
        JobData(job_name='test_job', categorical={'option1': 'value2'}),
        JobData(job_name='test_job'),
    ]

    for (query, jobs), expected in zip(small_db.iterate_database(), expected_queries, strict=True):
        assert query == expected
        expected_result = small_db.query(query)
        assert jobs == expected_result


def test_delete(small_db):
    """Test deletion of jobs with categorical where delete_all_children is False."""
    expected_result = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            memory=np.int64(100),
            runtime=np.int64(5),
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            numerical={"filesizes": np.array([123, 512, 128])},
            memory=np.int64(128),
            runtime=np.int64(6),
        ),
        JobData(
            job_name="test_job",
            slurm_id="3",
        ),
    ]

    simple_query = JobData(job_name="test_job")
    query_result = small_db.query(simple_query)
    assert query_result == expected_result

    small_db.delete(simple_query, delete_all_children=False)

    query_result = small_db.query(simple_query)
    assert query_result == []

    excepted_results = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            categorical={"option1": "value1", "option2": "value2"},
        ),
        JobData(
            job_name="test_job",
            slurm_id="4",
            runtime=7,
            memory=100,
            categorical={"option2": "value2", "option1": "value1"},
        ),
    ]

    query = JobData(
        job_name="test_job",
        categorical={"option1": "value1", "option2": "value2"},
    )
    query_result = small_db.query(query)

    assert query_result == excepted_results

    # Missing category values results to noop delete
    small_db.delete(
        JobData(job_name="test_job", categorical={"option1": "value1"}),
        delete_all_children=False,
    )

    query_result = small_db.query(query)

    assert query_result == excepted_results

    small_db.delete(query, delete_all_children=False)

    query_result = small_db.query(query)

    assert query_result == []


def test_delete_all_children(small_db):
    query_result = small_db.query(
        JobData(
            job_name="test_job", categorical={"option1": "value1", "option2": "value2"}
        )
    )
    assert len(query_result)

    small_db.delete(
        JobData(job_name="test_job", categorical={"option1": "value1"}),
        delete_all_children=True,
    )

    query_result = small_db.query(
        JobData(
            job_name="test_job", categorical={"option1": "value1", "option2": "value2"}
        )
    )
    assert query_result == []

    query_result = small_db.query(JobData(job_name="test_job"))
    assert len(query_result)

    query_result = small_db.query(
        JobData(job_name="test_job", categorical={"option1": "value2"})
    )
    assert len(query_result)
    small_db.delete(JobData(job_name="test_job"), delete_all_children=True)

    query_result = small_db.query(JobData(job_name="test_job"))
    assert query_result == []

    query_result = small_db.query(
        JobData(job_name="test_job", categorical={"option1": "value2"})
    )
    assert query_result == []


def test_update_missing_mem_elapsed(empty_h5py_file, monkeypatch):
    def mock_parse_slurm_job_metadata(slurm_id, step_id):
        return {
            "max_rss": 101,
            "elapsed_seconds": 100,
        }

    monkeypatch.setattr(
        "slurmise.job_database.slurm.parse_slurm_job_metadata",
        mock_parse_slurm_job_metadata,
    )

    with JobDatabase.get_database(empty_h5py_file) as db:
        db.record(
            JobData(
                job_name="test_job",
                slurm_id="1.0",
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="2.1",
                memory=128,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="3.extern",
                runtime=111,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        db.record(
            JobData(
                job_name="test_job",
                slurm_id="4.extern",
                memory=138,
                runtime=222,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": [123, 512, 128]},
            )
        )

        expected_output = [
            JobData(
                job_name="test_job",
                slurm_id="1.0",
                memory=101,
                runtime=100,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": np.array([123, 512, 128])},
            ),
            JobData(
                job_name="test_job",
                slurm_id="2.1",
                memory=128,
                runtime=100,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": np.array([123, 512, 128])},
            ),
            JobData(
                job_name="test_job",
                slurm_id="3.extern",
                runtime=111,
                memory=101,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": np.array([123, 512, 128])},
            ),
            JobData(
                job_name="test_job",
                slurm_id="4.extern",
                memory=138,
                runtime=222,
                categorical={"option1": "value1", "option2": "value2"},
                numerical={"filesizes": np.array([123, 512, 128])},
            ),
        ]

        results = db.query(
            JobData(
                job_name="test_job",
                categorical={"option1": "value1", "option2": "value2"},
            ),
            update_missing=True,
        )
        assert results == expected_output

        results = db.query(
            JobData(
                job_name="test_job",
                categorical={"option1": "value1", "option2": "value2"},
            ),
            update_missing=False,
        )
        assert results == expected_output
