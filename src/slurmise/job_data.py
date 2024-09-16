from dataclasses import dataclass, field, astuple
import numpy as np

def array_safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if a is b:
        return True
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    if isinstance(a, dict) and isinstance(b, dict):
        return (
            a.keys() == b.keys() and 
            all((a[key] == b[key]).all() for key in a.keys())
        )
    try:
        return a == b
    except TypeError:
        return NotImplemented


def dc_eq(dc1, dc2) -> bool:
    """checks if two dataclasses which hold numpy arrays are equal"""
    if dc1 is dc2:
        return True
    if dc1.__class__ is not dc2.__class__:
        return NotImplemented  # better than False
    t1 = astuple(dc1)
    t2 = astuple(dc2)
    return all(array_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


@dataclass(eq=False)
class JobData:
    job_name: str
    slurm_id: str
    categorical: dict = field(default_factory=lambda: {})
    numerical: dict = field(default_factory=lambda: {})
    memory: int | None = None
    runtime: int | None = None

    @staticmethod
    def from_dataset(job_name, slurm_id, dataset):
        runtime = dataset.get('runtime', None)[()]
        memory = dataset.get('memory', None)[()]
        numerical = {
            key: value[()]
            for key, value in dataset.items()
            if key not in ('runtime', 'memory')
        }

        return JobData(
            job_name=job_name,
            slurm_id=slurm_id,
            numerical=numerical,
            memory=memory,
            runtime=runtime,
        )

    def __eq__(self, other):
            return dc_eq(self, other)
