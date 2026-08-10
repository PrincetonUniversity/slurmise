#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich"]
# ///
"""Walks you through a lesson's README.md, running its commands as you go.

A lesson is any subdirectory holding a `README.md`. That file is the tutorial
-- the narration, the commands, and what each command is expected to do -- and
this drives it: renders the prose, shows each `$` command before running it, and
checks the command did what the markdown says it should. Commands run in the
lesson's own directory, so its `slurmise.toml` and `../bin/...` resolve.

    ./tutorial.py                  # pick a lesson from the menu, then walk it
    ./tutorial.py 02_jobs_in_loop  # skip the menu
    ./tutorial.py --yes            # every lesson, in order, unattended (CI)
    ./tutorial.py --option mock    # answer every either/or block with "mock"

Every lesson starts from a clean database -- its `#> reset` block runs first,
always -- so lessons can be taken in any order, or one on its own. Ctrl-C stops;
there is no resuming.

Expectations are the `#>` lines in the fences -- shell comments, so they're
inert if you copy a block and paste it into your own shell:

    $ some-command --flag
    #> expect /a regex the combined stdout+stderr must match/ retry=20 delay=4

    #> expect ok              exit 0, output unchecked (the default)
    #> expect fail            must exit non-zero
    #> reset                  (first line of a fence) run once, up front
    #> option <name>          opens one of several ways to do the same thing

`retry=` is how the markdown says "we're waiting on the cluster". Anything else
that doesn't match its expectation stops the tour loudly.

`#> option` is how a lesson offers a choice -- typically the same step done on
the cluster or faked locally. Each option runs until the next `#> option` or the
end of the fence:

    #> option cluster
    $ sbatch --wait run_thing.sbatch
    #> expect ok
    #> option mock
    $ bash mock_thing.sh
    #> expect ok

You pick one; `--option <name>` picks for you, and `--yes` alone takes the
first. The names mean nothing here -- they are the lesson's words, printed back
to whoever is choosing. Whether the options really are interchangeable, so the
rest of the lesson holds either way, is the lesson author's problem.

The shebang runs this under `uv`, which supplies `rich`. It does NOT need
`slurmise` importable -- but the commands it runs do, under whatever `python3`
is on your PATH.
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


def child_env() -> dict[str, str]:
    """The environment the tutorial's commands run in.

    `uv run --script` puts its ephemeral env (which has `rich` but not
    `slurmise`) first on PATH, so a `#!/usr/bin/env python3` script like
    `./slrmise` would pick up uv's interpreter instead of the user's. Strip that
    entry back out: the commands should see the shell the reader started from,
    exactly as if they had typed them.
    """
    env = os.environ.copy()
    if "UV_RUN_RECURSION_DEPTH" in env and sys.prefix != sys.base_prefix:
        ours = str(pathlib.Path(sys.prefix) / "bin")
        env["PATH"] = os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep) if p != ours)
        if env.get("VIRTUAL_ENV") == sys.prefix:
            env.pop("VIRTUAL_ENV", None)
    return env


# An `export` on its own line in a fence, ahead of the command it applies to.
# Written unglued because that is what a person would type; the parser folds it
# onto the following command so that one `bash -c` sees both.
ENV_LINE_RE = re.compile(r"^export\s+\w+=")


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
class Option:
    """One of several interchangeable ways to do the same step.

    `name` is whatever the markdown wrote after `#> option`, or None for a fence
    that offered no choice at all. It is never interpreted -- only shown to
    whoever is choosing, and matched against `--option`.
    """

    name: str | None = None
    steps: list[Step] = field(default_factory=list)


@dataclass
class Block:
    """A fenced block: one or more alternatives, of which exactly one is run."""

    options: list[Option] = field(default_factory=lambda: [Option()])
    is_reset: bool = False

    @property
    def offers_choice(self) -> bool:
        return len(self.options) > 1

    def pick(self, name: str | None) -> Option:
        """The option called `name`, else the first. Used when nobody is asked."""
        if name is not None:
            for option in self.options:
                if option.name == name:
                    return option
        return self.options[0]


@dataclass
class Section:
    id: str | None  # "01".."08" for a numbered section, else None
    title: str
    body: list[str | Block] = field(default_factory=list)  # prose | code


@dataclass
class Lesson:
    """A directory holding a `README.md`, and that file already parsed."""

    path: pathlib.Path
    title: str  # the markdown's `# Heading`, for the menu
    sections: list[Section]

    @property
    def name(self) -> str:
        return self.path.name


def find_lessons(root: pathlib.Path) -> list[Lesson]:
    """Every `<dir>/README.md` under `root`, in directory-name order.

    Which lessons exist depends on where this is run from: a generated tutorial
    ships only the numbered getting-started lessons, while a clone also has the
    work-in-progress ones that `generate_tutorial.py` excludes.
    """
    return [_load_lesson(path.parent) for path in sorted(root.glob("*/README.md"))]


def _load_lesson(path: pathlib.Path) -> Lesson:
    markdown = path / "README.md"
    title = next((line[2:].strip() for line in markdown.read_text().splitlines() if line.startswith("# ")), "")
    try:
        sections = parse_walkthrough(markdown)
    except ValueError as exc:  # say which file, now that there are several
        raise ValueError(f"{path.name}/README.md: {exc}") from exc
    return Lesson(path=path, title=title, sections=sections)


# ---------------------------------------------------------------------------
# parsing README.md
# ---------------------------------------------------------------------------

HEADING_RE = re.compile(r"^##\s+(?:(\d\d)\s*[—–-]\s*)?(.+?)\s*$")
DIRECTIVE_RE = re.compile(r"^#>\s*(expect|option|reset)\b\s*(.*)$")
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

    An `export ...` line is the exception: it belongs to the command below it,
    and is folded onto the front of it so both reach the same shell. Each command
    runs in its own `bash -c`, so an export left as a step of its own would
    evaporate before the command that needs it -- but a reader's shell keeps it,
    so the markdown must show it on its own line the way they'd type it, not
    welded on with a backslash to suit this script.

    `#> option <name>` starts a new alternative within the fence; the commands
    that follow belong to it until the next one.
    """
    sections: list[Section] = [Section(id=None, title="", body=[])]
    prose: list[str] = []
    block: Block | None = None
    # A command continued across `\` lines is kept as the authored lines, not
    # folded into one: bash runs it either way, and the tour can then show the
    # command broken exactly where the markdown breaks it.
    pending: list[str] | None = None
    env_prefix: list[str] = []  # `export` lines awaiting their command

    def flush_prose() -> None:
        text = "\n".join(prose).strip("\n")
        if text.strip():
            sections[-1].body.append(text)
        prose.clear()

    def add_step(lines: list[str]) -> None:
        """Record a command, carrying any `export` lines in front of it."""
        block.options[-1].steps.append(Step("\n".join(env_prefix + lines)))
        env_prefix.clear()

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
                env_prefix.clear()
                continue
            prose.append(line)
            continue

        # --- inside a fence ---
        if stripped.startswith("```"):
            if pending is not None:
                add_step(pending)
            if any(option.steps for option in block.options) or block.is_reset:
                sections[-1].body.append(block)
            block, pending = None, None
            env_prefix.clear()
            continue

        if pending is not None:  # continuation of a `\`-terminated command
            pending.append(line)
            if not stripped.endswith("\\"):
                add_step(pending)
                pending = None
            continue

        directive = DIRECTIVE_RE.match(stripped)
        if directive:
            name, arg = directive.group(1), directive.group(2)
            if name == "reset":
                block.is_reset = True
            elif name == "expect":
                if not block.options[-1].steps:
                    raise ValueError(f"`#> expect` with no command before it: {stripped}")
                block.options[-1].steps[-1].expect = _parse_expect(arg)
            else:  # option
                if not arg.strip():
                    raise ValueError("`#> option` needs a name")
                # The first one replaces the unnamed default, so a fence that
                # opens with `#> option` doesn't carry an empty option around.
                if block.options[-1].steps or block.options[-1].name:
                    block.options.append(Option(name=arg.strip()))
                else:
                    block.options[-1] = Option(name=arg.strip())
                pending = None
                env_prefix.clear()
            continue

        if stripped.startswith("$ ") or stripped == "$":
            cmd = stripped[2:] if stripped.startswith("$ ") else ""
            if cmd.endswith("\\"):
                pending = [cmd]
            elif cmd:
                add_step([cmd])
            continue

        if ENV_LINE_RE.match(stripped):
            env_prefix.append(stripped)
            continue
        # any other non-`$`, non-`#>` fence line is an output annotation -> ignored

    flush_prose()
    return [s for s in sections if s.body]


# ---------------------------------------------------------------------------
# presentation + execution
# ---------------------------------------------------------------------------


class Tour:
    def __init__(
        self,
        console: Console,
        lesson: Lesson,
        assume_yes: bool,
        option: str | None = None,
    ):
        self.console = console
        self.lesson = lesson
        self.cwd = lesson.path  # commands run here, not next to this script
        self.assume_yes = assume_yes
        self.option = option  # answer every `#> option` block with this name
        self.env = child_env()
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

    def choose_option(self, block: Block) -> Option:
        """Ask which of a block's alternatives to run.

        Asked every time a block offers a choice, rather than remembered: the
        lesson author is the one promising they leave the same state, so nothing
        here should quietly carry an earlier answer into a later block.
        """
        if self.option is not None or self.assume_yes:
            chosen = block.pick(self.option)
            self.console.print(f"\n[dim]option:[/dim] {chosen.name}")
            return chosen

        self.console.print()
        width = max(len(option.name or "") for option in block.options)
        for number, option in enumerate(block.options, start=1):
            first = option.steps[0].command.splitlines()[0] if option.steps else ""
            more = " …" if len(option.steps) > 1 or "\n" in option.steps[0].command else ""
            name = (option.name or "").ljust(width)
            self.console.print(f"  [bold]{number}[/bold]) [cyan]{name}[/cyan]  [dim]{first}{more}[/dim]")

        while True:
            try:
                answer = self.console.input("[dim]Choose ▸ [/dim]").strip()
            except EOFError:  # non-tty without --yes: behave like --yes
                self.assume_yes = True
                return block.options[0]
            if answer.isdigit() and 1 <= int(answer) <= len(block.options):
                return block.options[int(answer) - 1]
            by_name = [o for o in block.options if o.name == answer]
            if by_name:
                return by_name[0]
            self.console.print(f"[yellow]Not one of the choices:[/yellow] {answer}")

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

        Streaming matters: a command that waits on the cluster produces nothing
        for minutes, and capture_output() would make that look like a hang. A
        spinner covers the silence and disappears as soon as it says anything.

        What runs is exactly what the reader was shown -- there is no rewriting
        between the box and bash. A lesson that wants a command run differently
        says so with `#> option`.
        """
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=self.cwd,
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

    def run_step(self, step: Step, show: bool = True) -> str:
        """Run one command, honouring its `retry=`. Returns its output.

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
                return output

        self.failure(step.command, step.expect, reason, output)
        raise TourFailure(f"`{step.command}`")

    def run_block(self, block: Block) -> None:
        """Run one of a block's alternatives, start to finish."""
        if block.is_reset:
            return  # already run, up front, by run_reset()

        option = self.choose_option(block) if block.offers_choice else block.options[0]
        for step in option.steps:
            self.run_step(step)

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


def run_reset(tour: Tour) -> None:
    """Clear the lesson's database and models so it starts from nothing.

    Unconditional, and per lesson rather than once up front: a lesson only holds
    from a clean slate, and running it here is what keeps every lesson
    independent of the ones before it.
    """
    for section in tour.lesson.sections:
        for block in section.body:
            if isinstance(block, Block) and block.is_reset:
                for step in block.options[0].steps:
                    tour.run_step(step, show=False)
    tour.console.print("[dim]Cleared previous runs — starting from an empty database.[/dim]")


def choose_lessons(console: Console, lessons: list[Lesson], assume_yes: bool) -> list[Lesson]:
    """Ask which lesson to walk. Everything, in order, if there's no one to ask."""
    if assume_yes or not sys.stdin.isatty():
        return lessons

    console.print()
    console.print("[bold cyan]Lessons in this tutorial[/bold cyan]")
    width = max(len(lesson.name) for lesson in lessons)
    for number, lesson in enumerate(lessons, start=1):
        console.print(f"  [bold]{number}[/bold]) {lesson.name:<{width}}  [dim]{lesson.title}[/dim]")
    console.print("  [bold]a[/bold]) all of them, in order")
    console.print()

    while True:
        try:
            answer = console.input("[dim]Choose ▸ [/dim]").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []
        if answer in ("a", "all"):
            return lessons
        if answer.isdigit() and 1 <= int(answer) <= len(lessons):
            return [lessons[int(answer) - 1]]
        by_name = [lesson for lesson in lessons if lesson.name == answer]
        if by_name:
            return by_name
        console.print(f"[yellow]Not one of the choices:[/yellow] {answer}")


def run_lesson(console: Console, lesson: Lesson, args: argparse.Namespace) -> int:
    """Walk one lesson end to end. Returns a process exit code."""
    console.print()
    console.print(Rule(f"[bold]{lesson.name}[/bold] — {lesson.title}", style="white"))

    tour = Tour(console, lesson, assume_yes=args.yes, option=args.option)
    run_reset(tour)

    for section in (s for s in lesson.sections if s.id):
        try:
            tour.run_section(section)
        except KeyboardInterrupt:
            console.print(f"\n[dim]Stopped. ./tutorial.py {lesson.name} starts again from the top.[/dim]")
            return 130
        except TourFailure as exc:
            console.print(f"\n[bold red]{lesson.name} section {section.id} failed:[/bold red] {exc}")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "lessons",
        nargs="*",
        metavar="LESSON",
        help="Lesson directories to walk. Default: ask, or all of them under --yes.",
    )
    parser.add_argument("--yes", action="store_true", help="Run everything without pausing (CI).")
    parser.add_argument(
        "--option",
        metavar="NAME",
        help="Answer every `#> option` block with this name, instead of asking.",
    )
    args = parser.parse_args()

    # With no terminal to fit (CI, or piped to a file) rich assumes 80 columns,
    # which is narrower than the tutorial's own output. Pick a width that fits it.
    console = Console(
        highlight=False,
        theme=PROSE_THEME,
        width=None if sys.stdout.isatty() else 140,
    )

    try:
        lessons = find_lessons(HERE)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2
    if not lessons:
        console.print(f"[bold red]No lessons found:[/bold red] no */README.md under {HERE}")
        return 2

    if args.lessons:
        by_name = {lesson.name: lesson for lesson in lessons}
        unknown = [name for name in args.lessons if name.rstrip("/") not in by_name]
        if unknown:
            console.print(f"[bold red]No such lesson:[/bold red] {', '.join(unknown)}")
            console.print(f"[dim]Available: {', '.join(by_name)}[/dim]")
            return 2
        chosen = [by_name[name.rstrip("/")] for name in args.lessons]
    else:
        chosen = choose_lessons(console, lessons, assume_yes=args.yes)
        if not chosen:  # Ctrl-C at the menu
            return 0

    for lesson in chosen:
        code = run_lesson(console, lesson, args)
        if code:
            return 0 if code == 130 else code

    console.print()
    console.print(Rule("[bold green]tour complete[/bold green]", style="green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
