# AI-Generated Commit Messages

This repository includes a git hook that optionally generates commit messages using GitHub Copilot CLI.

## How It Works

When you run `git commit` (without `-m`), the hook checks if GitHub Copilot CLI is available. If it is, it generates a commit message in conventional commits format. If not, it does nothing and lets you write the message manually.

## Setup

The hook is automatically installed when you run:
```bash
uv run pre-commit install --hook-type prepare-commit-msg
```

## Prerequisites (Optional)

To enable AI-generated messages, install GitHub Copilot CLI:

```bash
# Via Homebrew
brew install copilot-cli

# Or via npm
npm install -g @githubnext/github-copilot-cli

# Authenticate
gh auth login
```

If Copilot CLI is not installed, the hook silently does nothing.

## Example

```bash
$ git add design/HYBRID.md
$ git commit

# If copilot is available, editor opens with:
# docs: Add hybrid HNSW-IVF index design documentation
#
# Documents architecture, concurrency control, and performance
# characteristics of the hybrid index implementation.
#
# Otherwise, editor opens with empty message template
```

## Notes

- The message is just a suggestion - edit before finalizing
- Hook only runs for `git commit` (not `-m`, `--amend`, merge, etc.)
- Skip with `git commit --no-verify` if needed
- Uninstall with `uv run pre-commit uninstall --hook-type prepare-commit-msg`

