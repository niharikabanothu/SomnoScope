#!/usr/bin/env bash
# One-shot: turn this folder into a git repo and push it to a new GitHub remote.
#
#   bash scripts/init_git.sh <github-username> [repo-name]
#
# Requires either the GitHub CLI (`gh`) -- in which case the remote repo is
# created for you -- or an existing empty repo you created in the browser.

set -euo pipefail

USER_NAME="${1:-}"
REPO="${2:-somnoscope}"

if [[ -z "$USER_NAME" ]]; then
  echo "usage: bash scripts/init_git.sh <github-username> [repo-name]" >&2
  exit 1
fi

if [[ -d .git ]]; then
  echo "this is already a git repo; skipping init"
else
  git init -b main
fi

git add .
git commit -m "SomnoScope: sleep staging with robustness and spectral explainability audit" \
  || echo "nothing to commit"

if command -v gh >/dev/null 2>&1; then
  gh repo create "$USER_NAME/$REPO" --public --source=. --remote=origin --push
else
  echo
  echo "gh CLI not found. Create an EMPTY repo at https://github.com/new named '$REPO', then:"
  echo "  git remote add origin git@github.com:$USER_NAME/$REPO.git"
  echo "  git push -u origin main"
fi
