#!/usr/bin/env sh
# Create the project virtualenv (outside the repo, for speed on WSL /mnt/c checkouts) and install pinned deps.
# Usage: scripts/setup_env.sh [venv_dir]
set -eu

VENV_DIR="${1:-$HOME/.venvs/miat-pi}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -r "$REPO_DIR/requirements.txt"
python -m pip install -e "$REPO_DIR[dev]"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "Activate with: . $VENV_DIR/bin/activate"
