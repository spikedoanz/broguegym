# AGENTS.md

## Local References

`docs/papers/` is intentionally untracked local reference material for agent work. Use files there
when they help with design or implementation context, but do not stage, commit, delete, or relocate
them unless the user explicitly asks.

## Pull Request Readiness

When a PR is approximately ready, run the local gate before marking it ready:

```bash
uv run pytest
uv run pyright
```

If using an independent `ycodex` review near the end, keep it bounded and treat non-completion as
non-blocking. Address straightforward findings; forward larger design comments to the user.

## Review Handoff

When iterating with another Codex session in tmux, finish each review-fix round by pushing the
branch, then send that session a concise re-review request with the PR number, branch/head, base,
and validation already run. Ask it to ping the current session back when done with either
actionable findings or `none`, rather than requiring constant polling.

When receiving a review handoff from another session, treat it as a read-only code review unless
explicitly told otherwise. Review the stated PR/base, prioritize correctness and experimental
control, report findings with severity and file/line references, and ping the requesting tmux
session back when done.
