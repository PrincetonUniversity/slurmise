# AI Contribution Policy

AI-assisted contributions are welcome in this project. We ask that contributors:

- **Disclose** that AI was used and name the tool and model (see commit trailer below).
- **Review and understand** every line submitted; the contributor is responsible for all code.
- **Meet the same quality, testing, and style standards** as any human contribution.
- **Not use fully autonomous agents** to open unsolicited issues or pull requests.
- **Respond to reviewers personally**; maintainers should not be interacting with bots.
- **Write your own PR description** rather than submitting a generated summary.

Only humans may be named as co-authors. AI tools may never sign off on a commit.

## Commit attribution

Use the Linux kernel convention to disclose AI assistance as a commit trailer:

```
Assisted-by: <harness>:<model>
```

For example:

```
Assisted-by: Claude Code:claude-sonnet-4-6
```

This policy follows the [Scientific Python AI contribution guide](https://learn.scientific-python.org/development/guides/ai/).
