import pytest

from slurmise.fit import MODEL_REGISTRY, model_factory


def test_model_factory():
    for name, cls in MODEL_REGISTRY.items():
        model_cls = model_factory(name)
        assert model_cls is cls

    with pytest.raises(ValueError, match="Unknown model: 'unknown'"):
        model_factory("unknown")
