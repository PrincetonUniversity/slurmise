import numpy as np
import pandas as pd
import pytest

from slurmise.job_data import JobData
from slurmise.utils import jobs_to_pandas


def test_jobs_to_pandas_basic():
    """Test basic conversion of JobData objects to pandas DataFrame."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check that runtime and memory are in the DataFrame
    assert "runtime" in df.columns
    assert "memory" in df.columns

    # Check that job_name, slurm_id, and cmd are not in the DataFrame
    assert "job_name" not in df.columns
    assert "slurm_id" not in df.columns
    assert "cmd" not in df.columns

    # Check the values
    assert df["runtime"].tolist() == [5, 6]
    assert df["memory"].tolist() == [100, 128]

    # Check that there are no categories or numerics
    assert categories == []
    assert numerics == []


def test_jobs_to_pandas_with_categories():
    """Test conversion with category features."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            categories={"option1": "value1", "option2": "value2"},
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
            categories={"option1": "value2", "option2": "value2"},
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check that category columns are present and have correct type
    assert "option1" in df.columns
    assert "option2" in df.columns
    assert df["option1"].dtype.name == "category"
    assert df["option2"].dtype.name == "category"

    # Check the categories list
    assert "option1" in categories
    assert "option2" in categories

    # Check that numerics is empty
    assert numerics == []

    # Check the values
    assert df["option1"].tolist() == ["value1", "value2"]
    assert df["option2"].tolist() == ["value2", "value2"]


def test_jobs_to_pandas_with_scalar_numerics():
    """Test conversion with scalar numeric features."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            numerics={"repetitions": 4, "iterations": 100},
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
            numerics={"repetitions": 8, "iterations": 200},
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check that numeric columns are present
    assert "repetitions" in df.columns
    assert "iterations" in df.columns

    # Check the numerics list
    assert "iterations" in numerics
    assert "repetitions" in numerics

    # Check that categories is empty
    assert categories == []

    # Check the values
    assert df["repetitions"].tolist() == [4, 8]
    assert df["iterations"].tolist() == [100, 200]


def test_jobs_to_pandas_with_numpy_array_numerics():
    """Test conversion with numpy array numeric features of equal size."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            numerics={"filesizes": np.array([123, 512, 128])},
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
            numerics={"filesizes": np.array([456, 789, 234])},
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check that numpy array is expanded into multiple columns
    assert "filesizes_0" in df.columns
    assert "filesizes_1" in df.columns
    assert "filesizes_2" in df.columns
    assert "filesizes" not in df.columns

    # Check the numerics list contains the expanded columns
    assert "filesizes_0" in numerics
    assert "filesizes_1" in numerics
    assert "filesizes_2" in numerics

    # Check that categories is empty
    assert categories == []

    # Check the values
    assert df["filesizes_0"].tolist() == [123, 456]
    assert df["filesizes_1"].tolist() == [512, 789]
    assert df["filesizes_2"].tolist() == [128, 234]


def test_jobs_to_pandas_with_mixed_features():
    """Test conversion with a mix of categories, scalar numerics, and array numerics."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            categories={"option1": "value1"},
            numerics={"repetitions": 4, "filesizes": np.array([123, 512])},
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
            categories={"option1": "value2"},
            numerics={"repetitions": 8, "filesizes": np.array([456, 789])},
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check categories
    assert "option1" in df.columns
    assert df["option1"].dtype.name == "category"
    assert categories == ["option1"]

    # Check scalar numerics
    assert "repetitions" in df.columns
    assert "repetitions" in numerics

    # Check array numerics
    assert "filesizes_0" in df.columns
    assert "filesizes_1" in df.columns
    assert "filesizes_0" in numerics
    assert "filesizes_1" in numerics

    # Check that columns are sorted
    assert df.columns.tolist() == sorted(df.columns.tolist())


def test_jobs_to_pandas_unequal_numpy_arrays():
    """Test that conversion fails when numpy arrays have different sizes."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            numerics={"filesizes": np.array([123, 512, 128])},
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
            numerics={"filesizes": np.array([456, 789])},  # Different size
        ),
    ]

    with pytest.raises(ValueError, match="numpy array of different sizes"):
        jobs_to_pandas(jobs)


def test_jobs_to_pandas_mixed_types_in_numeric():
    """Test that conversion fails when numeric columns have mixed types."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            numerics={"value": np.array([123, 512])},
        ),
        JobData(
            job_name="test_job",
            slurm_id="2",
            runtime=6,
            memory=128,
            numerics={"value": 456},  # Scalar instead of array
        ),
    ]

    with pytest.raises(ValueError, match="Numerics columns must be scalars or equal length numpy arrays"):
        jobs_to_pandas(jobs)


def test_jobs_to_pandas_empty_list():
    """Test conversion with an empty list of jobs."""
    jobs = []

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check that the DataFrame is empty
    assert df.empty
    assert categories == []
    assert numerics == []


def test_jobs_to_pandas_column_sorting():
    """Test that columns are sorted consistently."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            categories={"zebra": "a", "alpha": "b"},
            numerics={"xyz": 1, "abc": 2},
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check that columns are sorted
    assert df.columns.tolist() == sorted(df.columns.tolist())

    # Check that categories and numerics are sorted
    assert categories == sorted(categories)
    assert numerics == sorted(numerics)


def test_jobs_to_pandas_return_types():
    """Test that the return types are correct."""
    jobs = [
        JobData(
            job_name="test_job",
            slurm_id="1",
            runtime=5,
            memory=100,
            categories={"option1": "value1"},
            numerics={"threads": 4},
        ),
    ]

    df, categories, numerics = jobs_to_pandas(jobs)

    # Check return types
    assert isinstance(df, pd.DataFrame)
    assert isinstance(categories, list)
    assert isinstance(numerics, list)
    assert all(isinstance(cat, str) for cat in categories)
    assert all(isinstance(num, str) for num in numerics)
