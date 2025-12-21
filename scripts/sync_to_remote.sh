#!/bin/bash
# sync_to_remote.sh - Sync local code to remote A100 server
# Usage: ./sync_to_remote.sh [username]

set -e

# SSH config uses default settings, just use host directly
REMOTE_HOST="front.convergence.lip6.fr"
REMOTE_DIR="~/exg_train"


# Get the project root directory (parent of the scripts directory)
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing code from ${LOCAL_DIR} to ${REMOTE_HOST}:${REMOTE_DIR}..."

rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'output_eng' \
    --exclude 'checkpoints' \
    --exclude '*.png' \
    --exclude '*.csv' \
    --exclude '*.npy' \
    --exclude 'physionet.org' \
    "${LOCAL_DIR}/" \
    "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Done! Code synced to ${REMOTE_DIR} on remote."
echo ""
echo "Next steps on remote:"
echo "  ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "  cd ${REMOTE_DIR}"
echo "  source .venv/bin/activate"
