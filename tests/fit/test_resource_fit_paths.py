from pathlib import Path

import pytest

from slurmise.fit.poly_fit import PolynomialFit
from slurmise.fit.resource_fit import ResourceFit
from slurmise.job_data import JobData


@pytest.fixture(autouse=True)
def monkey_patch_basepath(tmp_path, monkeypatch):
    """Keep the default model path out of the user's home directory during testing."""
    monkeypatch.setattr("slurmise.fit.resource_fit.BASEMODELPATH", tmp_path)
    yield
    monkeypatch.undo()


def test_model_info_hash_ignores_category_order():
    """The hash must not depend on the insertion order of categories.

    The update path rebuilds categories from HDF5 group names, which JobDatabase writes
    sorted, while the predict path fills them in regex match order. If the hash were
    order dependent the two paths would resolve to different model directories and a
    prediction would silently never find the model that was just fit.
    """
    sorted_order = JobData(job_name="test_job", categories={"option1": "value1", "option2": "value2"})
    regex_order = JobData(job_name="test_job", categories={"option2": "value2", "option1": "value1"})

    assert ResourceFit._get_model_info_hash(sorted_order) == ResourceFit._get_model_info_hash(regex_order)
    assert ResourceFit._make_model_path(sorted_order) == ResourceFit._make_model_path(regex_order)


def test_model_info_hash_is_unique_per_job_and_categories():
    """Distinct job names, category values, and model classes hash apart (#76)."""
    base = JobData(job_name="test_job", categories={"option1": "value1"})
    other_job = JobData(job_name="other_job", categories={"option1": "value1"})
    other_value = JobData(job_name="test_job", categories={"option1": "value2"})
    extra_category = JobData(job_name="test_job", categories={"option1": "value1", "option2": "value2"})
    no_categories = JobData(job_name="test_job")

    hashes = [
        ResourceFit._get_model_info_hash(query)
        for query in (base, other_job, other_value, extra_category, no_categories)
    ]
    assert len(set(hashes)) == len(hashes)

    # The model class is part of the hash, so two model types never share a directory.
    assert ResourceFit._get_model_info_hash(base) != PolynomialFit._get_model_info_hash(base)


def test_make_model_path_roots_under_base_path(tmp_path):
    """A caller supplying only a base directory still gets a unique per-query subpath."""
    query = JobData(job_name="test_job", categories={"option1": "value1"})
    base_path = tmp_path / "slurmise_dir"

    model_path = PolynomialFit._make_model_path(query, base_path=base_path)

    assert model_path == base_path / "PolynomialFit" / PolynomialFit._get_model_info_hash(query)


def test_make_model_path_defaults_to_basemodelpath(tmp_path):
    """Without a base path the module level BASEMODELPATH is used (patched to tmp_path)."""
    query = JobData(job_name="test_job")

    model_path = PolynomialFit._make_model_path(query)

    assert model_path == tmp_path / "PolynomialFit" / PolynomialFit._get_model_info_hash(query)


def test_categories_do_not_share_a_model_directory(tmp_path):
    """Two categories of one job save to different directories instead of clobbering.

    This is the regression test for #76: before the fix every query resolved to the
    single configured base directory, so each fit overwrote the previous one.
    """
    base_path = tmp_path / "slurmise_dir"
    queries = [
        JobData(job_name="test_job", categories={"option1": "value1", "option2": "value2"}),
        JobData(job_name="test_job", categories={"option1": "value2"}),
        JobData(job_name="test_job"),
    ]

    paths = [PolynomialFit._make_model_path(query, base_path=base_path) for query in queries]

    assert len(set(paths)) == len(paths)
    for path in paths:
        assert base_path in Path(path).parents
