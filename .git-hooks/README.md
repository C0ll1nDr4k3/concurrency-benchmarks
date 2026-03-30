# AI-Generated Commit Messages

This repository includes a git hook that automatically generates commit messages using GitHub CLI's Copilot based on your staged changes.

## How It Works

When you run `git commit` (without `-m`), the hook uses `gh copilot suggest -t git` to analyze your staged changes and prepopulate the commit message editor with an AI-generated message in conventional commits format.

If `gh` CLI is not available or fails, the hook gracefully does nothing and lets you write the message manually.

## Setup

The hook is automatically installed when you run:
```bash
uv run pre-commit install --hook-type prepare-commit-msg
```

This is already done if you've installed pre-commit hooks for this repo.

## Prerequisites

You need the GitHub CLI with Copilot access:
```bash
# Install gh CLI if you don't have it
brew install gh  # macOS
# or follow: https://cli.github.com/

# Authenticate
gh auth login

# Copilot should work automatically if you have access
```

## Examples

```bash
$ git add src/hnsw_coarse_optimistic.hpp
$ git commit

# Editor opens with AI-generated message:
# fix: Add missing mutex header include
#
# Adds #include <mutex> to resolve std::unique_lock compilation errors
```

## Customization

The commit message is just a suggestion - edit it before finalizing. The hook only runs when:
- You run `git commit` without `-m` flag
- The commit message file is empty (no amend, merge, etc.)
- There are staged changes
- `gh` CLI is available

## Disabling

To skip the hook for a single commit:
```bash
git commit --no-verify
```

To uninstall:
```bash
uv run pre-commit uninstall --hook-type prepare-commit-msg
```
