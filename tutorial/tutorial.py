#!/usr/bin/env python3
"""Walks you through a lesson's README.md, running its commands as you go.

A lesson is any subdirectory holding a `README.md`. That file is the tutorial
-- the narration, the commands, and what each command is expected to do -- and
this drives it: renders the prose, shows each `$` command before running it, and
checks the command did what the markdown says it should. Commands run in the
lesson's own directory, so its `slurmise.toml` and `../bin/...` resolve.

    ./tutorial.py                  # pick a lesson from the menu, then walk it
    ./tutorial.py 02_jobs_in_loop  # skip the menu
    ./tutorial.py --yes            # every lesson, in order, unattended (CI)
    ./tutorial.py --mock           # take the no-cluster path throughout
    ./tutorial.py clean            # delete what walking the lessons produced

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

You pick one; `--yes` alone takes the first, and `--mock` takes the one named
`mock`. That name is the one exception to these being the lesson's own words,
meaningless here: `--mock` is how a reader with no cluster -- and CI, which has
none either -- asks for the faked path across every lesson at once. A block
offering a choice with no `mock` in it stops the tour under `--mock`, rather
than quietly submitting to a scheduler that isn't there. Whether the options
really are interchangeable, so the rest of the lesson holds either way, is the
lesson author's problem.

Nothing outside the standard library is needed to run this. It does NOT need
`slurmise` importable -- but the commands it runs do, under whatever `python3`
is on your PATH, which is also the one running this script.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# the terminal
# ---------------------------------------------------------------------------

# ANSI when someone is watching, empty strings when this is a pipe or a CI log
# -- there the escapes are noise nobody renders.
_TTY = sys.stdout.isatty()


def _seq(code: str) -> str:
    return code if _TTY else ""


BOLD = _seq("\033[1m")
DIM = _seq("\033[2m")
CYAN = _seq("\033[36m")
GREEN = _seq("\033[32m")
RED = _seq("\033[31m")
YELLOW = _seq("\033[33m")
OFF = _seq("\033[0m")

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def visible(text: str) -> int:
    """How wide `text` prints, ignoring the escapes that take no space."""
    return len(ANSI_RE.sub("", text))


def width() -> int:
    """How wide to draw.

    Off a terminal there is nothing to measure and 80 is the conventional guess,
    but the tutorial's own output -- `display`'s table especially -- is wider
    than that, so a piped run is given the room its content needs instead.
    """
    return shutil.get_terminal_size(fallback=(140, 24)).columns if _TTY else 140


def rule(label: str, color: str = "") -> None:
    """A horizontal divider with `label` sitting in it."""
    room = max(width() - visible(label) - 2, 0)
    left = room // 2
    print(f"{color}{'─' * left}{OFF} {label} {color}{'─' * (room - left)}{OFF}")


# A framed block: a titled top rule, a `│` gutter, a bottom rule. Split into its
# three parts because a command's output is framed as it streams, line by line,
# with no way to know up front how much of it there will be -- or whether there
# will be any at all.
#
# Deliberately open on the right. What goes inside is a command someone is meant
# to read and retype, or the output they are meant to compare against the
# README, and a closing border would mean either rewrapping that -- at a column
# that has nothing to do with the content -- or cutting it. Left open, an
# over-long line simply wraps past the frame and nothing is ever lost.


def box_top(color: str = "", title: str = "") -> None:
    head = f"─ {title} " if title else "──"
    print(f"{color}╭{head}{'─' * max(width() - visible(head) - 1, 0)}{OFF}")


def box_line(text: str, color: str = "") -> None:
    print(f"{color}│{OFF} {text}", flush=True)


def box_bottom(color: str = "") -> None:
    print(f"{color}╰{'─' * max(width() - 1, 0)}{OFF}")


def box(lines: list[str], color: str = "", title: str = "") -> None:
    box_top(color, title)
    for line in lines:
        box_line(line, color)
    box_bottom(color)


# The single `#> option` name this script knows by heart. Every other name is
# the lesson's own business -- shown in the menu, never acted on. This one is
# spelled out here so `--mock` means the same thing in every lesson at once.
MOCK = "mock"


# `code` and **bold** -- all the inline markdown the lessons actually use.
CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")


def prose(text: str) -> None:
    """Print a paragraph of the lesson's markdown, indented.

    Lines go out as authored rather than reflowed to the terminal. The READMEs
    are already wrapped to about 80 columns, and their prose carries
    four-space-indented code blocks and `|---` tree art that only reflowing
    could break. So the only thing done here is inline styling.
    """
    print()
    for line in text.splitlines():
        styled = CODE_RE.sub(rf"{CYAN}\1{OFF}", line)
        styled = BOLD_RE.sub(rf"{BOLD}\1{OFF}", styled)
        print(f"  {styled}" if line.strip() else "")


def ask(prompt: str) -> str:
    """Read a line, with the prompt dimmed. Raises EOFError with nobody there."""
    return input(f"{DIM}{prompt}{OFF}")


class Ticker:
    """`running… 12s`, rewritten in place while a command says nothing.

    A step that waits on the cluster produces no output for minutes, and without
    this that is indistinguishable from a hang. It stops at the command's first
    line of output, so anything talkative never shows it at all. Only ever run
    for someone watching a terminal -- rewriting a line means nothing to a log.
    """

    def __init__(self):
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._tick, daemon=True)

    def _tick(self) -> None:
        start = time.monotonic()
        while not self._done.wait(1.0):
            print(f"\r{DIM}  running… {time.monotonic() - start:.0f}s{OFF}", end="", flush=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop, and wipe the line so whatever prints next starts clean."""
        if self._done.is_set():
            return
        self._done.set()
        self._thread.join()
        print("\r\033[K", end="", flush=True)


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
    that offered no choice at all. Only shown to whoever is choosing, except for
    the one name `--mock` looks for.
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

    def named(self, name: str) -> Option | None:
        """The option called `name`, if this block offers one at all."""
        return next((option for option in self.options if option.name == name), None)


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
    """Every `<dir>/README.md` under `root`, in directory-name order."""
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
        lesson: Lesson,
        assume_yes: bool,
        mock: bool = False,
    ):
        self.lesson = lesson
        self.cwd = lesson.path  # commands run here, not next to this script
        self.assume_yes = assume_yes
        self.mock = mock  # answer every `#> option` block with the mock one

    # -- presentation --------------------------------------------------

    def section_banner(self, section: Section) -> None:
        label = f"{section.id} — {section.title}" if section.id else section.title
        print()
        rule(f"{BOLD}{CYAN}{label}{OFF}", CYAN)

    def present_command(self, cmd: str) -> None:
        """Show the command, then wait for Enter (Ctrl-C stops the tour).

        A command written across several `\\`-continued lines in the markdown is
        shown the same way here, so the box matches what you'd read and type.
        """
        print()
        box(cmd.splitlines(), GREEN, f"{DIM}${OFF}")
        if self.assume_yes:
            return
        try:
            ask("Press Enter to run it ▸ ")
        except EOFError:  # non-tty without --yes: behave like --yes
            self.assume_yes = True

    def choose_option(self, block: Block) -> Option:
        """Ask which of a block's alternatives to run.

        Asked every time a block offers a choice, rather than remembered: the
        lesson author is the one promising they leave the same state, so nothing
        here should quietly carry an earlier answer into a later block.
        """
        if self.mock:
            chosen = block.named(MOCK)
            if chosen is None:
                offered = ", ".join(option.name or "?" for option in block.options)
                raise TourFailure(f"--mock, but this block offers only: {offered}")
            print(f"\n{DIM}option:{OFF} {chosen.name}")
            return chosen

        if self.assume_yes:
            chosen = block.options[0]
            print(f"\n{DIM}option:{OFF} {chosen.name}")
            return chosen

        print()
        label_width = max(len(option.name or "") for option in block.options)
        for number, option in enumerate(block.options, start=1):
            first = option.steps[0].command.splitlines()[0] if option.steps else ""
            more = " …" if option.steps and (len(option.steps) > 1 or "\n" in option.steps[0].command) else ""
            name = (option.name or "").ljust(label_width)
            print(f"  {BOLD}{number}{OFF}) {CYAN}{name}{OFF}  {DIM}{first}{more}{OFF}")

        while True:
            try:
                answer = ask("Choose ▸ ").strip()
            except EOFError:  # non-tty without --yes: behave like --yes
                self.assume_yes = True
                return block.options[0]
            if answer.isdigit() and 1 <= int(answer) <= len(block.options):
                return block.options[int(answer) - 1]
            by_name = [o for o in block.options if o.name == answer]
            if by_name:
                return by_name[0]
            print(f"{YELLOW}Not one of the choices:{OFF} {answer}")

    def failure(self, cmd: str, expect: Expect, reason: str, output: str) -> None:
        commands = cmd.splitlines() or [cmd]
        tail = output.splitlines()[-15:] or ["(no output)"]
        print()
        box(
            [
                f"{BOLD}command {OFF} {CYAN}{commands[0]}{OFF}",
                *(f"         {CYAN}{line}{OFF}" for line in commands[1:]),
                f"{BOLD}expected{OFF} {expect.describe()}",
                f"{BOLD}but     {OFF} {YELLOW}{reason}{OFF}",
                "",
                f"{BOLD}{DIM}last lines of output{OFF}",
                *(f"{DIM}{line}{OFF}" for line in tail),
            ],
            RED,
            f"{BOLD}{RED}expectation not met{OFF}",
        )

    # -- execution -----------------------------------------------------

    def run_command(self, cmd: str) -> tuple[int, str]:
        """Run under bash, streaming output live while capturing it.

        Streaming matters: a command that waits on the cluster produces nothing
        for minutes, and capture_output() would make that look like a hang. A
        ticker covers the silence and disappears as soon as it says anything.

        Output is framed as it arrives, the same way the command above it was, so
        the pair reads as one exchange. Only the frame is added: the text inside
        is the command's own, unwrapped and untruncated, and it is what the `#>`
        expectations match against.

        What runs is exactly what the reader was shown -- there is no rewriting
        between the box and bash. A lesson that wants a command run differently
        says so with `#> option`. It runs in the environment the reader started
        from, untouched, since this script needs nothing added to it.
        """
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=self.cwd,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        ticker = Ticker() if _TTY and not self.assume_yes else None
        if ticker:
            ticker.start()
        opened = False  # the output frame is drawn lazily -- many commands are silent
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if ticker:
                    ticker.stop()
                    ticker = None
                if not opened:
                    box_top(DIM, f"{DIM}output{OFF}")
                    opened = True
                lines.append(line)
                # Colour the command emitted is passed through to a terminal and
                # stripped from a pipe -- what every colour-aware tool does, and
                # what keeps a CI log free of escapes nothing there will render.
                box_line((line if _TTY else ANSI_RE.sub("", line)).rstrip("\n"), DIM)
        finally:
            if ticker:
                ticker.stop()
            if opened:
                box_bottom(DIM)
        return proc.wait(), "".join(lines)

    def run_step(self, step: Step, show: bool = True) -> str:
        """Run one command, honouring its `retry=`. Returns its output.

        `show=False` runs it as housekeeping -- no box, no pause.
        """
        if show:
            self.present_command(step.command)
        for attempt in range(step.expect.retry + 1):
            if attempt:
                print(f"{DIM}not there yet — retry {attempt}/{step.expect.retry} in {step.expect.delay:g}s{OFF}")
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
                prose(item)
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
    print(f"{DIM}Cleared previous runs — starting from an empty database.{OFF}")


def choose_lessons(lessons: list[Lesson], assume_yes: bool) -> list[Lesson]:
    """Ask which lesson to walk. Everything, in order, if there's no one to ask."""
    if assume_yes or not sys.stdin.isatty():
        return lessons

    print()
    print(f"{BOLD}{CYAN}Lessons in this tutorial{OFF}")
    name_width = max(len(lesson.name) for lesson in lessons)
    for number, lesson in enumerate(lessons, start=1):
        print(f"  {BOLD}{number}{OFF}) {lesson.name:<{name_width}}  {DIM}{lesson.title}{OFF}")
    print(f"  {BOLD}a{OFF}) all of them, in order")
    print()

    while True:
        try:
            answer = ask("Choose ▸ ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return []
        if answer in ("a", "all"):
            return lessons
        if answer.isdigit() and 1 <= int(answer) <= len(lessons):
            return [lessons[int(answer) - 1]]
        by_name = [lesson for lesson in lessons if lesson.name == answer]
        if by_name:
            return by_name
        print(f"{YELLOW}Not one of the choices:{OFF} {answer}")


def run_lesson(lesson: Lesson, args: argparse.Namespace) -> int:
    """Walk one lesson end to end. Returns a process exit code."""
    print()
    rule(f"{BOLD}{lesson.name}{OFF} — {lesson.title}")

    tour = Tour(lesson, assume_yes=args.yes, mock=args.mock)
    run_reset(tour)

    for section in (s for s in lesson.sections if s.id):
        try:
            tour.run_section(section)
        except KeyboardInterrupt:
            print(f"\n{DIM}Stopped. ./tutorial.py {lesson.name} starts again from the top.{OFF}")
            return 130
        except TourFailure as exc:
            print(f"\n{BOLD}{RED}{lesson.name} section {section.id} failed:{OFF} {exc}")
            return 1
    return 0


# What walking a lesson leaves behind: its database, its fits, and whatever
# SLURM wrote next to it. All of it is gitignored, so `clean` is what gets the
# tree back to the state the release tarball ships in.
ARTIFACTS = (
    "*.h5",
    "*.pkl",
    "slurm*.out",
    "fits.json",
    "local.sql",
    "__pycache__",
    "slurm_outs",
    "out_slurm_logs",
)


def clean(lessons: list[Lesson]) -> int:
    """Delete the run artifacts under each of `lessons`."""
    removed = 0
    for lesson in lessons:
        for pattern in ARTIFACTS:
            for path in sorted(lesson.path.rglob(pattern)):
                if not path.exists():
                    continue  # an earlier pattern already took the directory holding it
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                print(f"{DIM}removed{OFF} {path.relative_to(HERE)}")
                removed += 1
    print(f"cleaned {removed} path(s) under {len(lessons)} lesson(s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "lessons",
        nargs="*",
        metavar="LESSON",
        help="Lesson directories to walk, or `clean` to delete what walking them produced. "
        "Default: ask, or all of them under --yes.",
    )
    parser.add_argument("--yes", action="store_true", help="Run everything without pausing (CI).")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Take the no-cluster path at every `#> option` block, instead of asking.",
    )
    args = parser.parse_args()

    try:
        lessons = find_lessons(HERE)
    except ValueError as exc:
        print(f"{BOLD}{RED}{exc}{OFF}")
        return 2
    if not lessons:
        print(f"{BOLD}{RED}No lessons found:{OFF} no */README.md under {HERE}")
        return 2

    # `clean` as the first word is the one thing that isn't a lesson name. No
    # lesson directory can collide with it: a lesson is `<dir>/README.md`, and
    # `clean/` would be a lesson called "clean", which the tutorial has no
    # reason to grow.
    wanted, cleaning = args.lessons, False
    if wanted and wanted[0].rstrip("/") == "clean":
        wanted, cleaning = wanted[1:], True

    if wanted:
        by_name = {lesson.name: lesson for lesson in lessons}
        unknown = [name for name in wanted if name.rstrip("/") not in by_name]
        if unknown:
            print(f"{BOLD}{RED}No such lesson:{OFF} {', '.join(unknown)}")
            print(f"{DIM}Available: {', '.join(by_name)}{OFF}")
            return 2
        chosen = [by_name[name.rstrip("/")] for name in wanted]
    elif cleaning:
        chosen = lessons  # `clean` with no lesson named cleans all of them
    else:
        chosen = choose_lessons(lessons, assume_yes=args.yes)
        if not chosen:  # Ctrl-C at the menu
            return 0

    if cleaning:
        return clean(chosen)

    for lesson in chosen:
        code = run_lesson(lesson, args)
        if code:
            return 0 if code == 130 else code

    print()
    rule(f"{BOLD}{GREEN}tour complete{OFF}", GREEN)
    return 0


if __name__ == "__main__":
    sys.exit(main())
