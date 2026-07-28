#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
"""Walks you through tutorial.md, running its commands as you go.

The tutorial itself -- the narration, the commands, and what each command is
expected to do -- lives entirely in tutorial.md. This drives it: renders the
prose, shows each `$` command before running it, and checks the command did what
the markdown says it should.

    ./tutorial.py                # interactive: shows each command, pauses, runs it
    ./tutorial.py --yes          # run the whole tour unattended (CI)
    ./tutorial.py --mock         # SLRMISE_MOCK=1: no scheduler, whole tour in <1 min

Every run starts from a clean database -- the `#> reset` block runs first, always.
Ctrl-C stops; there is no resuming, because `--mock --yes` takes under a minute.

Expectations are the `#>` lines in the fences -- shell comments, so they're
inert if you copy a block and paste it into your own shell:

    $ some-command --flag
    #> expect /a regex the combined stdout+stderr must match/ retry=20 delay=4

    #> expect ok              exit 0, output unchecked (the default)
    #> expect fail            must exit non-zero
    #> repeat-until /re/ max=8    (last line of a fence) re-run the whole block
    #> reset                  (first line of a fence) run once, up front

`retry=`/`repeat-until` is how the markdown says "we're waiting on the cluster".
Anything else that doesn't match its expectation stops the tour loudly.

The shebang runs this under `uv`, which supplies `rich`. It does NOT need
`slurmise` importable -- but the commands it runs (`./slrmise`) do, under
whatever `python3` is on your PATH.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

# Rich renders `inline code` as bold cyan *on black*. On any terminal whose
# background isn't black that reads as a dark blob, and the background bleeds
# across line wraps. Same colour, no background.
PROSE_THEME = Theme({"markdown.code": "cyan", "markdown.code_block": "cyan"})

HERE = pathlib.Path(__file__).resolve().parent
TUTORIAL_MD = HERE / "tutorial.md"


def child_env(mock: bool = False) -> dict[str, str]:
    """The environment the tutorial's commands run in.

    `uv run --script` puts its ephemeral env (which has `rich` but not
    `slurmise`) first on PATH, so a `#!/usr/bin/env python3` script like
    `./slrmise` would pick up uv's interpreter instead of the user's. Strip that
    entry back out: the commands should see the shell the reader started from,
    exactly as if they had typed them.

    `--mock` is just `SLRMISE_MOCK=1` -- slrmise itself does the pretending, so
    you can equally well export it and type the commands by hand.
    """
    env = os.environ.copy()
    if "UV_RUN_RECURSION_DEPTH" in env and sys.prefix != sys.base_prefix:
        ours = str(pathlib.Path(sys.prefix) / "bin")
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != ours)
        if env.get("VIRTUAL_ENV") == sys.prefix:
            env.pop("VIRTUAL_ENV", None)
    if mock:
        env["SLRMISE_MOCK"] = "1"
    return env


# ---------------------------------------------------------------------------
# the document model
# ---------------------------------------------------------------------------


@dataclass
class Expect:
    """What a command must do. `kind` is "ok", "fail", or "match"."""

    kind: str = "ok"
    pattern: re.Pattern[str] | None = None
    retry: int = 0  # extra attempts beyond the first
    delay: float = 4.0  # seconds between attempts

    def check(self, code: int, output: str) -> str | None:
        """Return None if satisfied, else a human-readable reason."""
        if self.kind == "fail":
            return None if code != 0 else "expected a non-zero exit, but it succeeded"
        if code != 0:
            return f"expected success, but it exited {code}"
        if self.kind == "match" and self.pattern and not self.pattern.search(output):
            return "it exited 0 but nothing in the output matched"
        return None

    def describe(self) -> str:
        if self.kind == "fail":
            return "a non-zero exit"
        if self.kind == "match" and self.pattern:
            return f"exit 0 and output matching /{self.pattern.pattern}/"
        return "exit 0"


@dataclass
class Step:
    command: str
    expect: Expect = field(default_factory=Expect)


@dataclass
class Block:
    """A fenced block: the commands in it plus its block-level directives."""

    steps: list[Step] = field(default_factory=list)
    repeat_until: re.Pattern[str] | None = None
    repeat_max: int = 8
    is_reset: bool = False


@dataclass
class Section:
    id: str | None  # "01".."08" for a numbered lesson, else None
    title: str
    body: list[str | Block] = field(default_factory=list)  # prose | code


# ---------------------------------------------------------------------------
# parsing tutorial.md
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^##\s+(?:(\d\d)\s*[—–-]\s*)?(.+?)\s*$")
DIRECTIVE_RE = re.compile(r"^#>\s*(expect|repeat-until|reset)\b\s*(.*)$")
ARG_RE = re.compile(r"^/(.*)/\s*(.*)$")  # /regex/ trailing key=value pairs


def _parse_options(text: str) -> dict[str, str]:
    return dict(tok.split("=", 1) for tok in text.split() if "=" in tok)


def _parse_expect(arg: str) -> Expect:
    """`fail` | `ok` | `/regex/ retry=N delay=S`"""
    word = arg.split()[0] if arg.split() else "ok"
    if word in ("ok", "fail"):
        return Expect(kind=word)
    match = ARG_RE.match(arg)
    if not match:
        raise ValueError(f"cannot parse `#> expect {arg}`")
    opts = _parse_options(match.group(2))
    return Expect(
        kind="match",
        pattern=re.compile(match.group(1)),
        retry=int(opts.get("retry", 0)),
        delay=float(opts.get("delay", 4)),
    )


def parse_walkthrough(path: pathlib.Path) -> list[Section]:
    """Turn the markdown into sections of prose and fenced command blocks.

    Inside a fence, `$ `-prefixed lines are commands (a trailing `\\` continues
    one onto the next line), `#>` lines are directives, and everything else is
    an output annotation for the reader and is ignored here.
    """
    sections: list[Section] = [Section(id=None, title="", body=[])]
    prose: list[str] = []
    block: Block | None = None
    # A command continued across `\` lines is kept as the authored lines, not
    # folded into one: bash runs it either way, and the tour can then show the
    # command broken exactly where the markdown breaks it.
    pending: list[str] | None = None

    def flush_prose() -> None:
        text = "\n".join(prose).strip("\n")
        if text.strip():
            sections[-1].body.append(text)
        prose.clear()

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if block is None:
            heading = HEADING_RE.match(line)
            if heading:
                flush_prose()
                sections.append(Section(id=heading.group(1), title=heading.group(2)))
                continue
            if stripped.startswith("```"):
                flush_prose()
                block = Block()
                pending = None
                continue
            prose.append(line)
            continue

        # --- inside a fence ---
        if stripped.startswith("```"):
            if pending is not None:
                block.steps.append(Step("\n".join(pending)))
            if block.steps or block.is_reset:
                sections[-1].body.append(block)
            block, pending = None, None
            continue

        if pending is not None:  # continuation of a `\`-terminated command
            pending.append(line)
            if not stripped.endswith("\\"):
                block.steps.append(Step("\n".join(pending)))
                pending = None
            continue

        directive = DIRECTIVE_RE.match(stripped)
        if directive:
            name, arg = directive.group(1), directive.group(2)
            if name == "reset":
                block.is_reset = True
            elif name == "expect":
                if not block.steps:
                    raise ValueError(f"`#> expect` with no command before it: {stripped}")
                block.steps[-1].expect = _parse_expect(arg)
            else:  # repeat-until
                match = ARG_RE.match(arg)
                if not match:
                    raise ValueError(f"cannot parse `#> repeat-until {arg}`")
                block.repeat_until = re.compile(match.group(1))
                block.repeat_max = int(_parse_options(match.group(2)).get("max", 8))
            continue

        if stripped.startswith("$ ") or stripped == "$":
            cmd = stripped[2:] if stripped.startswith("$ ") else ""
            if cmd.endswith("\\"):
                pending = [cmd]
            elif cmd:
                block.steps.append(Step(cmd))
        # non-`$`, non-`#>` fence lines are output annotations -> ignored

    flush_prose()
    return [s for s in sections if s.body]


# ---------------------------------------------------------------------------
# presentation + execution
# ---------------------------------------------------------------------------


class Tour:
    def __init__(self, console: Console, assume_yes: bool, mock: bool = False):
        self.console = console
        self.assume_yes = assume_yes
        self.env = child_env(mock)
        # Commands run on a pipe, where rich would assume 80 columns and squeeze
        # `display`'s table until it clips cells with an ellipsis -- unreadable,
        # and it would break the `#>` expectations that match on those values.
        # Give them room: the table sizes itself to its content, and on a narrow
        # terminal we'd rather the frame split a full row than lose characters.
        # Never below the width `display`'s table needs. Squeezing it makes rich
        # clip cells with an ellipsis, and a clipped value both reads badly and
        # silently defeats any `#>` expectation that matches on it. If the frame
        # is narrower than that, we would rather it split a full row than lose
        # characters.
        self.env["COLUMNS"] = str(max(console.width - 4, 120))
        if console.is_terminal:  # keep their colour across the pipe we capture on
            self.env["FORCE_COLOR"] = "1"

    # -- presentation --------------------------------------------------

    def section_banner(self, section: Section) -> None:
        label = f"{section.id} — {section.title}" if section.id else section.title
        self.console.print()
        self.console.print(Rule(f"[bold cyan]{label}[/bold cyan]", style="cyan"))

    def prose(self, text: str) -> None:
        self.console.print()
        self.console.print(Padding(Markdown(text), (0, 2)))

    def present_command(self, cmd: str) -> None:
        """Show the command, then wait for Enter (Ctrl-C stops the tour).

        A command written across several `\\`-continued lines in the markdown is
        shown the same way here, so the box matches what you'd read and type.
        """
        self.console.print()
        self.console.print(
            Panel(
                Syntax(cmd, "bash", background_color="default", word_wrap=True),
                title="[dim]$[/dim]",
                title_align="left",
                border_style="green",
                padding=(0, 1),
            )
        )
        if self.assume_yes:
            return
        try:
            self.console.input("[dim]Press Enter to run it ▸ [/dim]")
        except EOFError:  # non-tty without --yes: behave like --yes
            self.assume_yes = True

    def failure(self, cmd: str, expect: Expect, reason: str, output: str) -> None:
        tail = "\n".join(output.splitlines()[-15:]) or "(no output)"
        self.console.print()
        self.console.print(
            Panel(
                Text.assemble(
                    ("command  ", "bold"),
                    (cmd, "cyan"),
                    "\n",
                    ("expected ", "bold"),
                    expect.describe(),
                    "\n",
                    ("but      ", "bold"),
                    (reason, "yellow"),
                    "\n\n",
                    ("last lines of output\n", "bold dim"),
                    (tail, "dim"),
                ),
                title="[bold red]expectation not met[/bold red]",
                border_style="red",
            )
        )

    def frame_line(self, line: str) -> None:
        """Print one output line inside the `│` frame.

        Hard-split rather than word-wrapped, and split by us rather than by the
        terminal: a line left to soft-wrap would put its tail outside the frame.
        Splitting on width also keeps `display`'s columns in order.

        `from_ansi` keeps whatever colour the command emitted (the tour asks for
        it with FORCE_COLOR, since commands run on a pipe) -- slicing a Text
        carries the styles with it, where slicing the raw string would cut an
        escape sequence in half.
        """
        width = max(20, self.console.width - 4)  # 2 indent + "│ "
        text = Text.from_ansi(line)
        for start in range(0, max(len(text), 1), width):
            self.console.print("  [dim]│[/dim] ", end="")
            self.console.print(text[start : start + width], no_wrap=True, overflow="ignore")

    # -- execution -----------------------------------------------------

    def run_command(self, cmd: str) -> tuple[int, str]:
        """Run under bash, streaming output live while capturing it.

        Streaming matters: the `while squeue` waits produce nothing for minutes,
        and capture_output() would make them look like a hang. A spinner covers
        that silence and disappears as soon as the command says anything.
        """
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=HERE,
            executable="/bin/bash",
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        status = None if self.assume_yes else self.console.status("[dim]running…[/dim]")
        if status:
            status.start()
        opened = False  # the output frame is drawn lazily -- many commands are silent
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if status:
                    status.stop()
                    status = None
                if not opened:
                    self.console.print("  [dim]╭─[/dim]")
                    opened = True
                lines.append(line)
                self.frame_line(line.rstrip("\n"))
        finally:
            if status:
                status.stop()
            if opened:
                self.console.print("  [dim]╰─[/dim]")
        return proc.wait(), "".join(lines)

    def run_step(self, step: Step, strict: bool = True, show: bool = True) -> tuple[bool, str]:
        """Run one command, honouring its `retry=`. Returns (satisfied, output).

        `show=False` runs it as housekeeping -- no box, no pause.
        """
        if show:
            self.present_command(step.command)
        for attempt in range(step.expect.retry + 1):
            if attempt:
                self.console.print(
                    f"[dim]not there yet — retry {attempt}/{step.expect.retry} in {step.expect.delay:g}s[/dim]"
                )
                time.sleep(step.expect.delay)
            code, output = self.run_command(step.command)
            reason = step.expect.check(code, output)
            if reason is None:
                return True, output

        if strict:
            self.failure(step.command, step.expect, reason, output)
        return False, output

    def run_block(self, block: Block) -> None:
        """Run a block's commands, re-running the whole block while a
        `repeat-until` is unsatisfied. Inside such a block an unmet per-command
        expectation is not fatal either -- it just means "go round again"."""
        if block.is_reset:
            return  # already run, up front, by run_reset()

        looping = block.repeat_until is not None
        for attempt in range(block.repeat_max if looping else 1):
            if attempt:
                self.console.print(
                    f"[dim]not settled yet — running the block again ({attempt}/{block.repeat_max})[/dim]"
                )
            output, ok = "", True
            for step in block.steps:
                ok, output = self.run_step(step, strict=not looping)
                if not ok:
                    if not looping:
                        raise TourFailure(f"`{step.command}`")
                    break
            if not looping:
                return
            if ok and block.repeat_until.search(output):
                return

        raise TourFailure(f"block never reached /{block.repeat_until.pattern}/ in {block.repeat_max} attempts")

    def run_section(self, section: Section) -> None:
        self.section_banner(section)
        for item in section.body:
            if isinstance(item, str):
                self.prose(item)
            else:
                self.run_block(item)


class TourFailure(Exception):
    pass


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_reset(tour: Tour, sections: list[Section]) -> None:
    """Clear the database and models so the tour starts from nothing.

    Unconditional, because the lessons only hold from a clean slate: lesson 04
    can only be OUT_OF_MEMORY if there is no earlier success at that intensity
    for self-heal to have learned from.
    """
    for section in sections:
        for block in section.body:
            if isinstance(block, Block) and block.is_reset:
                for step in block.steps:
                    tour.run_step(step, show=False)
    tour.console.print("[dim]Cleared previous runs — starting from an empty database.[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="Run everything without pausing (CI).")
    parser.add_argument("--mock", action="store_true", help="SLRMISE_MOCK=1: pretend there's a scheduler.")
    args = parser.parse_args()

    # With no terminal to fit (CI, or piped to a file) rich assumes 80 columns,
    # which is narrower than the tutorial's own output. Pick a width that fits it.
    console = Console(
        highlight=False,
        theme=PROSE_THEME,
        width=None if sys.stdout.isatty() else 140,
    )
    os.chdir(HERE)  # so '.' base_dir and ../bin/... resolve on the shared fs
    (HERE / "out_slurm_logs").mkdir(exist_ok=True)
    os.environ.setdefault("SBATCH_OUTPUT", "out_slurm_logs/slurm-%j.out")

    try:
        sections = parse_walkthrough(TUTORIAL_MD)
    except ValueError as exc:
        console.print(f"[bold red]tutorial.md:[/bold red] {exc}")
        return 2

    tour = Tour(console, assume_yes=args.yes, mock=args.mock)
    if args.mock:
        console.print("[yellow]--mock:[/yellow] SLRMISE_MOCK=1 — no jobs will actually be submitted.")
    run_reset(tour, sections)

    for section in (s for s in sections if s.id):
        try:
            tour.run_section(section)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopped. ./tutorial.py starts again from the top.[/dim]")
            return 0
        except TourFailure as exc:
            console.print(f"\n[bold red]Lesson {section.id} failed:[/bold red] {exc}")
            return 1

    console.print()
    console.print(Rule("[bold green]tour complete[/bold green]", style="green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
