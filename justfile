# List the recipes
default:
    @just --list

# Build the docs into site/
docs-build:
    rm -rf .cache
    uv run --no-sync zensical build --clean --strict

# Preview the docs at http://localhost:8000 while editing
docs-serve:
    uv run --no-sync zensical serve

# Build the docs and publish them to GitHub Pages (gh-pages branch)
docs-deploy: _docs-check docs-build
    uvx ghp-import --no-jekyll --push --force \
        --message "Deploy docs for $(git describe --tags --always)" site

# Refuse to deploy docs that do not match the published sources
_docs-check:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! git diff --quiet HEAD; then
        echo "error: uncommitted changes, commit or stash them before deploying" >&2
        exit 1
    fi
    git fetch --quiet
    if [ -n "$(git rev-list HEAD..@{upstream} 2>/dev/null)" ]; then
        echo "error: $(git rev-parse --abbrev-ref HEAD) is behind its upstream, pull before deploying" >&2
        exit 1
    fi
    if ! uv run --no-sync python -c "import qseek_insights" 2>/dev/null; then
        echo "error: the docs need the qseek-insights plugin: uv pip install -e ../qseek-insights" >&2
        exit 1
    fi
