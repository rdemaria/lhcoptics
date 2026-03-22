#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [major|minor|bug]" >&2
  exit 1
}

BUMP=${1:-}
case "$BUMP" in
  major|minor|bug|patch) ;;
  *) usage ;;
esac
[ "$BUMP" = "patch" ] && BUMP="bug"

NAME=lhcoptics

if git status --porcelain | grep -q .; then
  echo "Working tree is dirty. Commit or stash changes before releasing." >&2
  exit 1
fi

CURRENT_VER=$(
  python - <<'PY'
import pathlib, re

init_py = pathlib.Path("src/lhcoptics/__init__.py")
match = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    init_py.read_text(),
    re.MULTILINE,
)
if not match:
    raise SystemExit(f"Could not find __version__ in {init_py}")
print(match.group(1))
PY
)

NEW_VER=$(
  python - "$BUMP" <<'PY'
import pathlib, re, sys

bump = sys.argv[1]
pyproj = pathlib.Path("pyproject.toml")
init_py = pathlib.Path("src/lhcoptics/__init__.py")

init_text = init_py.read_text()
match = re.search(
    r'^__version__\s*=\s*["\']([^"\']+)["\']',
    init_text,
    re.MULTILINE,
)
if not match:
    raise SystemExit(f"Could not find __version__ in {init_py}")
ver = match.group(1)
major, minor, patch = map(int, ver.split("."))
if bump == "major":
    major += 1
    minor = 0
    patch = 0
elif bump == "minor":
    minor += 1
    patch = 0
else:
    patch += 1
new_ver = f"{major}.{minor}.{patch}"

init_text = re.sub(
    r'(?m)^(__version__\s*=\s*["\'])[^"\']+(["\'])',
    lambda m: f"{m.group(1)}{new_ver}{m.group(2)}",
    init_text,
    count=1,
)
init_py.write_text(init_text)

pyproj_text = pyproj.read_text()
if re.search(r'(?m)^version\s*=\s*"[^"]+"', pyproj_text):
    pyproj_text = re.sub(
        r'(?m)^version\s*=\s*"[^"]+"',
        lambda _m: f'version = "{new_ver}"',
        pyproj_text,
        count=1,
    )
    pyproj.write_text(pyproj_text)

print(new_ver)
PY
)

echo "========================================================================"
echo "Preparing release: $NAME $CURRENT_VER -> $NEW_VER"
echo "========================================================================"

if git tag --list "v$NEW_VER" | grep -q .; then
  echo "Tag v$NEW_VER already exists. Aborting." >&2
  exit 1
fi

rm -rf dist/ *.egg-info

git add pyproject.toml src/lhcoptics/__init__.py
git commit -m "Release v$NEW_VER"

echo "Building distribution..."
python -m build

echo "Uploading to pypi..."
twine upload dist/*

echo "Tagging and pushing..."
git tag "v$NEW_VER"
git push origin HEAD
git push origin "v$NEW_VER"
