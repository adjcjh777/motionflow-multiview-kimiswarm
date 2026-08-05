#!/bin/bash
# Push the current branch to GitHub using the token stored in Git Credential Manager.
set -e

TOKEN=$(printf 'protocol=https\nhost=github.com\n' | git credential-manager get 2>/dev/null | grep '^password=' | sed 's/^password=//')
if [ -z "$TOKEN" ]; then
    echo "No GitHub token found in credential manager."
    exit 1
fi

REPO="https://adjcjh777:${TOKEN}@github.com/adjcjh777/motionflow-multiview-kimiswarm.git"
git remote set-url origin "$REPO"
git push -u origin "$(git rev-parse --abbrev-ref HEAD)" "$@"
