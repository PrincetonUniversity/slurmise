# slurmise tutorial

One directory per lesson. Each holds the lesson itself in `lesson.md`, plus the files
it operates on: a `slurmise.toml`, the `.sbatch` scripts, and the mock scripts that
stand in for a scheduler. The toy programs the lessons run are in `bin/`.

You can work through `lesson.md` as it is — it is Markdown, so the commands and the
explanation sit together — or read the rendered version, where every command has its
real output printed beneath it:

<https://princetonuniversity.github.io/slurmise/>

Those outputs are not pasted in. Every command in the book is executed when the book
is built, so what you see is what the command really produced, and a lesson that stops
working breaks the build.

There is no script that runs the lessons for you. That is deliberate: the point of the
exercise is to run the commands yourself and look at what comes back.
