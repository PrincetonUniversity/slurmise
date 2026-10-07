from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from slurmise.extras import snake_parsers
from slurmise.job_data import JobData
from slurmise.resource_corrector import ResourceCorrector


def test_input():
    sp = snake_parsers.SnakemakeV8()
    # no index, return first element
    no_index = sp.input()
    assert no_index(None, None, ["correct", "wrong"]) == "correct"

    # get index by number
    second_index = sp.input(1)
    assert second_index(None, None, ["wrong", "Correct", "wrong"]) == "Correct"

    # get index by name
    named_index = sp.input("request")
    assert named_index(None, None, {"request": "CORRECT"}) == "CORRECT"


def test_wildcards():
    sp = snake_parsers.SnakemakeV8()
    named_index = sp.wildcards("request")
    assert named_index(None, {"request": "CORRECT"}, None) == "CORRECT"


@dataclass
class DummyRule:
    resources: dict | None = None
    params: dict | None = None


def test_threads():
    sp = snake_parsers.SnakemakeV8()
    threads = sp.threads()

    # non_callable threads
    const_threads = DummyRule(resources={"_cores": 12})
    assert threads(const_threads, None, None) == 12

    # callable with only wildcards
    callable_threads = DummyRule(resources={"_cores": lambda wildcards: wildcards})
    assert threads(callable_threads, "wildcards", "input") == "wildcards"

    # callable with wildcards and input
    callable_threads = DummyRule(resources={"_cores": lambda wildcards, input: (wildcards, input)})
    assert threads(callable_threads, "wildcards", "input") == ("wildcards", "input")


def test_params():
    sp = snake_parsers.SnakemakeV8()
    params = sp.params("target")

    # non_callable params
    const_params = DummyRule(params={"target": "result"})
    assert params(const_params, None, None) == "result"

    # only wildcards
    callable_params = DummyRule(params={"target": lambda wildcards: wildcards})
    assert params(callable_params, "wc", "inpt") == "wc"

    # wildcards and input
    callable_params = DummyRule(params={"target": lambda wildcards, input: (wildcards, input)})
    assert params(callable_params, "wc", "inpt") == ("wc", "inpt")

    # invalid options and input
    match_result = f"Cannot use param {'target'!r} in slurmise.  Input functions may only depend on wildcards or input."

    callable_params = DummyRule(params={"target": lambda wildcards, output: wildcards})
    with pytest.raises(ValueError, match=match_result):
        params(callable_params, "wc", "inpt")

    callable_params = DummyRule(params={"target": lambda wildcards, threads: wildcards})
    with pytest.raises(ValueError, match=match_result):
        params(callable_params, "wc", "inpt")

    callable_params = DummyRule(params={"target": lambda wildcards, resources: wildcards})
    with pytest.raises(ValueError, match=match_result):
        params(callable_params, "wc", "inpt")


def test_build_variables():
    sp = snake_parsers.SnakemakeV8()
    with pytest.raises(ValueError, match="The wildcards source for no_key requires a key entry"):
        sp.build_variables({"no_key": "wildcards"})

    with pytest.raises(ValueError, match="The params source for no_key requires a key entry"):
        sp.build_variables({"no_key": "params"})

    parsers = sp.build_variables(
        {
            "input_var": "input",
            "input_var_key": ("input", 1),
            "wildcards_var": ("wildcards", "test_wc"),
            "threads_var": "threads",
            "params_var": ("params", "test_param"),
        }
    )

    rule = DummyRule(
        resources={"_cores": 4},
        params={"test_param": "tested param"},
    )
    wildcards = {"test_wc": "tested wildcards"}
    inputs = ["input0", "input1"]

    parsed = {var: parser(rule, wildcards, inputs) for var, parser in parsers.items()}

    assert parsed == {
        "input_var": "input0",
        "input_var_key": "input1",
        "wildcards_var": "tested wildcards",
        "threads_var": 4,
        "params_var": "tested param",
    }


# ---------------------------------------------------------------------------
# make_predictor
# ---------------------------------------------------------------------------


def _mock_slurmise(runtime=50.0, memory=2000.0):
    """Return a mock Slurmise that raw_predict returns fixed resource values."""
    mock = MagicMock()
    job_data = JobData(job_name="rule_name", runtime=runtime, memory=memory)

    def raw_predict(jd, attempt=0):

        rt_corrector = ResourceCorrector(resource="runtime", default=60.0, retry_exponent=1.0)
        mem_corrector = ResourceCorrector(resource="memory", default=1000.0, retry_exponent=1.0)
        jd.runtime, _ = rt_corrector.correct(runtime, False, "rule_name", attempt)
        jd.memory, _ = mem_corrector.correct(memory, False, "rule_name", attempt)
        return jd, []

    mock.raw_predict.side_effect = raw_predict
    mock.job_data_from_dict.return_value = job_data
    return mock


def test_make_predictor_returns_runtime():
    sp = snake_parsers.SnakemakeV8()
    slurmise = _mock_slurmise(runtime=45.0)
    rule = DummyRule(resources={"_cores": 1}, params={})
    rule.name = "myrule"
    predictor = sp.make_predictor(slurmise, {}, rule, "runtime")
    result = predictor(wildcards={}, input=[], attempt=1)
    assert result == pytest.approx(45.0)


def test_make_predictor_returns_memory():
    sp = snake_parsers.SnakemakeV8()
    slurmise = _mock_slurmise(memory=3000.0)
    rule = DummyRule(resources={"_cores": 1}, params={})
    rule.name = "myrule"
    predictor = sp.make_predictor(slurmise, {}, rule, "memory")
    result = predictor(wildcards={}, input=[], attempt=1)
    assert result == pytest.approx(3000.0)


def test_make_predictor_retry_scaling_via_corrector():
    """Retry scaling is handled by the ResourceCorrector inside raw_predict.

    attempt=2 with retry_exponent=1.0 (default) doubles the base prediction.
    """
    sp = snake_parsers.SnakemakeV8()
    slurmise = _mock_slurmise(runtime=50.0)
    rule = DummyRule(resources={"_cores": 1}, params={})
    rule.name = "myrule"
    predictor = sp.make_predictor(slurmise, {}, rule, "runtime")

    attempt1 = predictor(wildcards={}, input=[], attempt=1)
    attempt2 = predictor(wildcards={}, input=[], attempt=2)

    # attempt=1 → 50 * 1**1 = 50; attempt=2 → 50 * 2**1 = 100
    assert attempt1 == pytest.approx(50.0)
    assert attempt2 == pytest.approx(100.0)
