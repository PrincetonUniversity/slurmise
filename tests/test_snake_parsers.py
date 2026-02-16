from dataclasses import dataclass
import pytest

import slurmise.extras.snake_parsers as sp
from slurmise.job_data import JobData


def test_input():
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
    named_index = sp.wildcards("request")
    assert named_index(None, {"request": "CORRECT"}, None) == "CORRECT"


@dataclass
class DummyRule:
    resources: dict | None = None
    params: dict | None = None


def test_threads():
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
    match_result = (
        f'Cannot use param {"target"!r} in slurmise.  ' 'Input functions may only depend on wildcards or input.'
    )

    callable_params = DummyRule(params={"target": lambda wildcards, output: wildcards})
    with pytest.raises(ValueError, match=match_result):
        params(callable_params, "wc", "inpt")

    callable_params = DummyRule(params={"target": lambda wildcards, threads: wildcards})
    with pytest.raises(ValueError, match=match_result):
        params(callable_params, "wc", "inpt")

    callable_params = DummyRule(params={"target": lambda wildcards, resources: wildcards})
    with pytest.raises(ValueError, match=match_result):
        params(callable_params, "wc", "inpt")


def test_ThreadScaler_defaults():
    ts = sp.ThreadScaler(memory_per_thread=10)

    # memory too low, default to 1 thread
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 1
    assert result_jd.runtime == 100
    assert result_jd.memory == 1

    # memory too low, default to 1 thread
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 5)
    assert result_threads == 1
    assert result_jd.runtime == 500  # scaled by current threads, too high
    assert result_jd.memory == 1

    # memory too high, default to 20 threads
    jd = JobData(job_name="test", memory=1000, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 20
    assert result_jd.runtime == 5
    assert result_jd.memory == 1000

    # intermediate value
    jd = JobData(job_name="test", memory=45, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 5  # ceiling value
    assert result_jd.runtime == 20
    assert result_jd.memory == 45


def test_ThreadScaler_change_clip():
    ts = sp.ThreadScaler(memory_per_thread=10, thread_range=(2, 10))

    # memory too low, default to 2 threads
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 2
    assert result_jd.runtime == 50
    assert result_jd.memory == 1

    # memory too low, default to 2 threads
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 5)
    assert result_threads == 2
    assert result_jd.runtime == 250  # scaled by current threads, too high
    assert result_jd.memory == 1

    # memory too high, default to 10 threads
    jd = JobData(job_name="test", memory=1000, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 10
    assert result_jd.runtime == 10
    assert result_jd.memory == 1000

    # intermediate value
    jd = JobData(job_name="test", memory=45, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 5  # ceiling value
    assert result_jd.runtime == 20
    assert result_jd.memory == 45


def test_ThreadScaler_clip_overheads():
    # when the overhead is < 1, set to 1
    ts = sp.ThreadScaler(memory_per_thread=10, runtime_overhead=0.8, memory_overhead=-2)
    assert ts.runtime_overhead == 1
    assert ts.memory_overhead == 1


def test_ThreadScaler_linear_overheads():
    # when the overhead is >= 2, it's added as a value * the number of threads
    ts = sp.ThreadScaler(memory_per_thread=10, runtime_overhead=3, memory_overhead=2)

    # memory too low, default to 1 thread, no changes
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 1
    assert result_jd.runtime == 100
    assert result_jd.memory == 1

    # memory too low, default to 1 threads
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 5)
    assert result_threads == 1
    assert result_jd.runtime == 500  # scaled by current threads, too high
    assert result_jd.memory == 1

    # memory too high, default to 10 threads
    jd = JobData(job_name="test", memory=1000, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 20
    assert result_jd.runtime == 5 + 3 * 19
    assert result_jd.memory == 1000 + 2 * 19

    # intermediate value
    jd = JobData(job_name="test", memory=45, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 5  # ceiling value
    assert result_jd.runtime == 20 + 3 * 4
    assert result_jd.memory == 45 + 2 * 4


def test_ThreadScaler_exp_overheads():
    # when the overhead is < 2, it's multiplied as a value ** the number of threads
    ts = sp.ThreadScaler(memory_per_thread=10, runtime_overhead=1.1, memory_overhead=1.2)

    # memory too low, default to 1 thread, no changes
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 1
    assert result_jd.runtime == 100
    assert result_jd.memory == 1

    # memory too low, default to 1 threads
    jd = JobData(job_name="test", memory=1, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 5)
    assert result_threads == 1
    assert result_jd.runtime == 500  # scaled by current threads, too high
    assert result_jd.memory == 1

    # memory too high, default to 10 threads
    jd = JobData(job_name="test", memory=1000, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 20
    assert result_jd.runtime == int(5 * 1.1**19)  # 30
    assert result_jd.memory == int(1000 * 1.2**19)  # 31K

    # intermediate value
    jd = JobData(job_name="test", memory=45, runtime=100)
    result_jd, result_threads = ts.update_job_data(jd, 1)
    assert result_threads == 5  # ceiling value
    assert result_jd.runtime == int(20 * 1.1**4)  # 29
    assert result_jd.memory == int(45 * 1.2**4)  # 93
