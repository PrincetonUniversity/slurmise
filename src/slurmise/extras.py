from slurmise.api import Slurmise
from snakemake.utils import format
import shutil
import functools

# TODO: accept some customization on the threads, scaling with attempts, fudge factor
def patch_snakemake_workflow(slurmise: Slurmise, workflow, rules: list[str]):
    slurmise_path = shutil.which('slurmise')

    # TODO: this may be fragile, uses _onstart
    original_onstart = workflow._onstart
    def onstart_slurmise_update(log):
        print('updating all models')
        original_onstart(log)
        slurmise.update_all_models()


    workflow.onstart(onstart_slurmise_update)

    for rule_name in rules:
        breakpoint()
        rule = workflow.get_rule(rule_name)
        if not rule.is_shell:
            continue

        # update shell command to record
        # TODO: need to escape existing quotes
        commands = rule.shellcmd.split('\n')
        command = commands[-1]  # only use last command
        rule.shellcmd += (
            '\n'
            f'{slurmise_path} --toml {slurmise.toml_path} '
            f'record --job-name {rule.name} "{command}"'
        )

        def format_command(rule, **variables):
            '''Format command with wildcards and globals.
            Mostly from snakemake.jobs.format_wildcards.'''
            threads = rule.resources['_cores']
            if not isinstance(threads, int):
                threads = 1
            _variables = {}
            _variables.update(rule.workflow.globals)
            _variables.update(
                dict(
                    rule=rule.name,
                    rulename=rule.name,
                    bench_iteration=None,
                    threads=threads,
                )
            )
            _variables.update(variables)
            return format(command, **_variables)

        @functools.cache
        def get_job_data(command: str, job_name: str):
            '''Helper function to cache predict calls.'''
            return slurmise.predict(command, job_name)

        # update resources
        def slurmise_memory(wildcards, input, attempt):
            command = format_command(
                rule,
                input=input,
                wildcards=wildcards,
                output="output",
            )
            return get_job_data(command, rule.name).memory * attempt
        rule.resources['mem_mb'] = slurmise_memory

        def slurmise_runtime(wildcards, input, attempt):
            command = format_command(
                rule,
                input=input,
                wildcards=wildcards,
                output="output",
            )
            return get_job_data(command, rule.name).runtime * attempt
        rule.resources['runtime'] = slurmise_runtime

