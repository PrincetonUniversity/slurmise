from __future__ import annotations

from typing import TYPE_CHECKING

import snakemake
from packaging import version
from snakemake.logging import logger

from slurmise.api import Slurmise
from slurmise.extras import snake_parsers
from slurmise.job_data import JobData
from slurmise.job_parse.file_parsers import FileMD5

SLURMISE_DEFAULTS = {
    "attempt_exp": 1,
    "memory_scale": 1.1,
    "runtime_scale": 1.25,
}


def patch_snakemake_workflow(
    slurmise: Slurmise,
    workflow: Workflow,
    rules: dict[str, dict],
    benchmark_dir: str | Path = "slurmise/benchmarks",
    keep_benchmarks: bool = False,
    record_benchmarks: bool = True,
):
    benchmark_dir = Path(benchmark_dir)

    original_onstart = workflow._onstart

    def onstart_slurmise_update(log):
        original_onstart(log)
        logger.info("SLURMISE: Updating all models")
        slurmise.update_all_models()

    workflow.onstart(onstart_slurmise_update)

    original_onsuccess = workflow._onsuccess

    def onsuccess_slurmise_update(log):
        original_onsuccess(log)
        if not record_benchmarks:
            logger.info("SLURMISE: Skipping recording completed jobs")
            return
        logger.info("SLURMISE: Recording completed jobs")
        md5_parser = FileMD5()
        for file in benchmark_dir.rglob("*.jsonl"):
            benchmark_data = json.loads(file.read_text())
            slurmise_data = json.loads(benchmark_data["params"]["slurmise_data"])

            try:
                runtime = (float(benchmark_data["s"]) / 60,)
            except ValueError:
                runtime = 0
            try:
                memory = (float(benchmark_data["max_rss"]),)
            except ValueError:
                memory = 0

            # if a value is a thread, update it to true value
            slurmise_data = _correct_threads(slurmise_data, benchmark_data)

            job_data = JobData(
                job_name=benchmark_data["rule_name"],
                slurm_id=md5_parser.parse_file(file),
                categories=slurmise_data["categories"],
                numerics=slurmise_data["numerics"],
                runtime=runtime,
                memory=memory,
            )

            slurmise.raw_record(job_data)
        if not keep_benchmarks:
            shutil.rmtree(benchmark_dir)

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
