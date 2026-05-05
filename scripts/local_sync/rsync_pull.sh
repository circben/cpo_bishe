#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-user@remote-host}"
REMOTE_DIR="${REMOTE_DIR:-~/cpo_project/outputs/remote_inference/jobs}"
LOCAL_DIR="${LOCAL_DIR:-./outputs/remote_inference/jobs}"

rsync -av --partial --info=stats1 --exclude "*.tmp" "${REMOTE_HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
