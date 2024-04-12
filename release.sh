#!/bin/bash
set -euo pipefail; IFS=$'\n\t'

NAME=xlay
VER=$( python -c "import $NAME; print($NAME.__version__)" )

echo "========================================================================"
echo "Tagging $NAME v$VER"
echo "========================================================================"

git tag v$VER
git push origin v$VER

echo "========================================================================"
echo "Releasing $NAME v$VER on PyPI"
echo "========================================================================"

python -m build
twine upload dist/*
rm -r dist/ *.egg-info
