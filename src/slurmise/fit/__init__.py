from slurmise.fit.kneighbors_fit import KNNFit
from slurmise.fit.poly_fit import PolynomialFit
from slurmise.fit.resource_fit import ResourceFit

MODEL_REGISTRY = {
    "poly": PolynomialFit,
    "knn": KNNFit,
}


def model_factory(name: str) -> ResourceFit:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
