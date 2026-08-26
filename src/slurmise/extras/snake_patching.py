from __future__ import annotations

from typing import TYPE_CHECKING

import snakemake
from packaging import version
from snakemake.logging import logger

from slurmise.api import Slurmise
from slurmise.extras import snake_parsers

if TYPE_CHECKING:
    from snakemake.workflow import Workflow


def _make_patch(adapter: snake_parsers.SnakemakeAdapter):
    def patch(self, workflow: Workflow):
        return adapter.patch_snakemake_workflow(self, workflow)

    return patch


patching_fncs = {
    7: _make_patch(snake_parsers.SnakemakeV7()),
    8: _make_patch(snake_parsers.SnakemakeV8()),
    9: _make_patch(snake_parsers.SnakemakeV9()),
}

snakemake_version = version.parse(snakemake.__version__)
if snakemake_version.major < 7:
    raise ValueError("Slurmise only supports snakemake>=7.0")

elif snakemake_version.major not in patching_fncs:
    raise ValueError(f"Slurmise does not support snakemake version {snakemake_version}")

else:
    logger.info(f"SLURMISE: detected snakemake v{snakemake_version}")
    Slurmise.register_patch(patching_fncs[snakemake_version.major])
