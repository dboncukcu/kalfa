#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
    cat >&2 <<'TEXT'
usage: bash tools/release.sh VERSION [SUMMARY]

Writes VERSION into pyproject.toml and uv.lock, then asks before the commit,
the annotated tag and the push. Nothing is committed without a yes. It first
offers ruff check --fix, whose fixes join the release commit, and pytest.

  bash tools/release.sh 0.2.6
  bash tools/release.sh 0.2.6 "the std catalogue loads lazily"
TEXT
    exit 2
}

die() {
    printf 'release: %s\n' "$*" >&2
    exit 1
}

confirm() {
    local answer
    read -r -p "$1 [y/N] " answer || return 1
    [[ "$answer" == [yY] || "$answer" == [yY][eE][sS] ]]
}

[[ $# -ge 1 && $# -le 2 ]] || usage
NEW="$1"
SUMMARY="${2-}"

[[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc|\.post|\.dev)[0-9]+)?$ ]] || die "$NEW is no version, expected 0.2.6"
git rev-parse --git-dir >/dev/null 2>&1 || die "$ROOT is no git repository"

OLD="$(awk -F'"' '/^version = "/ { print $2; exit }' pyproject.toml)"
[[ -n "$OLD" ]] || die "pyproject.toml carries no version line"
[[ "$OLD" != "$NEW" ]] || die "pyproject.toml is at $NEW already"
if git rev-parse -q --verify "refs/tags/v$NEW" >/dev/null; then
    die "tag v$NEW exists already"
fi

printf '%s -> %s\n' "$OLD" "$NEW"

if confirm "run ruff --fix and pytest first?"; then
    FAILED=()
    uv run ruff check --fix src tests tools || FAILED+=("ruff")
    uv run pytest -q || FAILED+=("pytest")
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        printf 'release: %s reported problems\n' "${FAILED[*]}" >&2
        confirm "go on anyway?" || die "stopped, the version is untouched"
    fi
fi

TEMPORARY="$(mktemp)"
trap 'rm -f "$TEMPORARY"' EXIT
awk -v version="$NEW" '
    /^version = "/ && !done { sub(/"[^"]*"/, "\"" version "\""); done = 1 }
    { print }
' pyproject.toml >"$TEMPORARY"
cp "$TEMPORARY" pyproject.toml
printf 'pyproject.toml: version = "%s"\n' "$NEW"

if command -v uv >/dev/null 2>&1; then
    uv lock --quiet
    printf 'uv.lock: refreshed\n'
else
    printf 'release: uv is not on PATH, uv.lock still says %s\n' "$OLD" >&2
fi

if ! command -v uv >/dev/null 2>&1; then
    printf 'release: uv is not on PATH, DOCS.md still says %s\n' "$OLD" >&2
else
    printf 'DOCS.md is written from the registry again, in case it drifted\n'
    uv run kalfa docs --write DOCS.md >/dev/null
    REGENERATED=1
    printf 'DOCS.md: %s\n' "$NEW"
fi

git add pyproject.toml uv.lock
[[ -z "${REGENERATED-}" ]] || git add DOCS.md

LEFT="$(git status --porcelain | grep -v '^[MARCD] ' || true)"
if [[ -n "$LEFT" ]]; then
    printf '\nnot staged:\n%s\n\n' "$LEFT"
    if confirm "stage these too?"; then
        git add -A
    fi
fi

SUBJECT="kalfa $NEW"
[[ -z "$SUMMARY" ]] || SUBJECT="kalfa $NEW: $SUMMARY"

printf '\n'
git --no-pager diff --cached --stat
printf '\n'
confirm "commit as \"$SUBJECT\"?" || die "stopped, the version is set but nothing is committed"
git commit -q -m "$SUBJECT"
printf 'committed %s\n' "$(git rev-parse --short HEAD)"

if confirm "tag v$NEW?"; then
    git tag -a "v$NEW" -m "v$NEW"
    printf 'tagged v%s\n' "$NEW"
fi

if confirm "push the commit and the tag to origin?"; then
    git push origin HEAD
    if git rev-parse -q --verify "refs/tags/v$NEW" >/dev/null; then
        git push origin "v$NEW"
    fi
fi
