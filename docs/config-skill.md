# Building a configuration with an AI assistant

slurmise ships an [Agent Skill](https://agentskills.io) called `slurmise-config`. It
helps an AI coding assistant write or edit a `slurmise.toml` and then checks the
result with `slurmise validate`. These instructions cover
[Claude Code](https://claude.com/claude-code).

The skill is a convenience, not a replacement for understanding the file. Read
{doc}`configuration` first so you can judge what it produces.

## What the skill does

When you ask for help setting up slurmise, the assistant will:

1. Look for an existing `slurmise.toml`, in the working directory and then in
   `~/.slurmise/`. It edits a file it finds in place.
2. Ask for two or three real example commands with different values.
3. Ask whether you run the command from the CLI or from Snakemake, and what the job
   should be called.
4. Ask modelling questions: whether each varying value
   is numeric, a category, or ignored; which feature of each input file predicts cost
   (size, line count, a value read by `awk`, or the file name or checksum); and the
   runtime (minutes) and memory (MB) defaults and maximums. If you don't know the
   limits it leaves them out.
5. Draft the TOML, then run `slurmise validate` against the example commands and fix
   the errors until it passes. It shows you what was extracted from each command and
   asks you to confirm it is what you meant.
6. Report any warnings and explain them.

The skill never runs `record`, `update-model` or `predict`, so it does not read or
change your job database, and it does not edit anything other than the configuration.

## Installing the skill

The skill is the `skills/slurmise-config` directory of the slurmise repository. It is
not part of the pip package, so download it from GitHub and copy it into a skills
directory that Claude Code reads:

- `.claude/skills/` in a project, to make the skill available in that project only.
- `~/.claude/skills/` in your home directory, to make it available in every project.

For example, to install it for all of your projects:

```bash
git clone --depth 1 https://github.com/PrincetonUniversity/slurmise.git
mkdir -p ~/.claude/skills
cp -r slurmise/skills/slurmise-config ~/.claude/skills/
```

For a single project, run the last two commands from the project root instead,
copying into `.claude/skills/`.

The skill is written against the current development version of slurmise, and relies on
the `slurmise validate` command.

Make sure the `slurmise` command is installed and on your `PATH` in the environment
the assistant works in, because it runs `slurmise validate` to check its work. See
{doc}`install`.

## Using the skill

Ask for what you need in plain language, for example:

> Set up slurmise for `nupack monomer -T 4 -C high`.

or invoke the skill directly with `/slurmise-config`. You can also ask it to add a job to
an existing configuration, or to fix a configuration that fails to load.

Give it real commands, and have any files they reference present in the working
directory to run file parsers during validation.

## Reviewing what it wrote

Always read the generated file before you rely on it. In particular:

- Check the `parsed` output from `validate`. It shows the numeric and category values
  extracted from each of your commands, which is the quickest way to spot a value
  treated as the wrong type.
- Check that runtime limits are in minutes and memory limits are in MB.
- Run `slurmise validate` yourself on a few more commands than the assistant used.

Once the configuration looks right, continue with {doc}`tutorial` to record jobs and
fit models.
