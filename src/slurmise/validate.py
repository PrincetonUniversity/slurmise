from __future__ import annotations

import shutil
import subprocess
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from slurmise.config import SlurmiseConfiguration, find_config_file
from slurmise.job_parse.file_parsers import AwkParser


@dataclass
class ValidationReport:
    """Outcome of validating a configuration file."""

    toml_path: Path | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parsed: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "toml_path": None if self.toml_path is None else str(self.toml_path),
            "errors": self.errors,
            "warnings": self.warnings,
            "parsed": self.parsed,
        }

    def format(self) -> str:
        lines = [] if self.toml_path is None else [f"Configuration: {self.toml_path}"]
        for parsed in self.parsed:
            lines.append(
                f"Parsed {parsed['cmd']!r} as job {parsed['job_name']}: "
                f"numerics={parsed['numerics']}, categories={parsed['categories']}"
            )
        lines += [f"Warning: {warning}" for warning in self.warnings]
        lines += [f"Error: {error}" for error in self.errors]
        lines.append("Configuration is valid." if self.ok else "Configuration is NOT valid.")
        return "\n".join(lines)


def validate_config(
    toml_path: str | Path | None = None,
    commands: Sequence[str] = (),
    job_name: str | None = None,
) -> ValidationReport:
    """Load a configuration and parse example commands against it, collecting problems.

    Errors are conditions that make the configuration unusable.  Warnings are legal but
    likely unintended.  Without ``toml_path`` the usual search locations are used.
    """
    report = ValidationReport()

    try:
        report.toml_path = Path(toml_path) if toml_path is not None else find_config_file()
        config = SlurmiseConfiguration(report.toml_path, create_dirs=False)
    except tomllib.TOMLDecodeError as e:
        report.errors.append(f"Invalid TOML: {e}")
        return report
    except KeyError as e:
        report.errors.append(f"Missing required configuration key: {e}")
        return report
    except (RuntimeError, ValueError, OSError) as e:
        report.errors.append(str(e))
        return report

    _check_models(config, report)
    _check_warnings(config, report)

    for cmd in commands:
        try:
            parsed = config.parse_job_cmd(cmd, job_name)
        except (ValueError, OSError, subprocess.CalledProcessError) as e:
            report.errors.append(f"Command {cmd!r} failed to parse: {e}")
            continue
        report.parsed.append(
            {
                "cmd": cmd,
                "job_name": parsed.job_name,
                "numerics": parsed.numerics,
                "categories": parsed.categories,
            }
        )

    return report


def _check_models(config: SlurmiseConfiguration, report: ValidationReport) -> None:
    for job_name in config.job_configurations:
        try:
            config.get_model_class(job_name)
        except ValueError as e:
            report.errors.append(f"Job {job_name}: {e}")


def _check_warnings(config: SlurmiseConfiguration, report: ValidationReport) -> None:
    # Inference takes the first job whose prefix starts the command, so a later job
    # whose prefix begins with an earlier one's (or equals it) can never be inferred.
    names = list(config.job_prefixes)
    for i, earlier in enumerate(names):
        for later in names[i + 1 :]:
            if config.job_prefixes[later].startswith(config.job_prefixes[earlier]):
                report.warnings.append(
                    f"Job {later} (prefix {config.job_prefixes[later]!r}) is shadowed by job {earlier} "
                    f"(prefix {config.job_prefixes[earlier]!r}) when inferring job names from commands. "
                    "List it first or give the jobs distinct prefixes."
                )

    report.warnings += [
        f"File parser {name!r} has no awk_script and is ignored." for name in config.ignored_file_parsers
    ]
    report.warnings += [
        f"File parser {name!r} replaces the built-in parser of the same name."
        for name in config.overridden_file_parsers
    ]

    uses_awk = any(
        isinstance(parser, AwkParser)
        for job in config.job_configurations.values()
        for parsers in job["job_spec_obj"].file_parsers.values()
        for parser in parsers
    )
    if uses_awk and shutil.which("awk") is None:
        report.warnings.append("A job uses an awk file parser but `awk` was not found on PATH.")
