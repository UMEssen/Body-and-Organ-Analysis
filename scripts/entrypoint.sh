#!/bin/bash
set -e

DOCKER_USER="${DOCKER_USER:-1000:1000}"
USER_ID="${DOCKER_USER%:*}"
GROUP_ID="${DOCKER_USER#*:}"

for dir in /storage_directory /workspace; do
    if [ -d "$dir" ]; then
        chown -R "$USER_ID:$GROUP_ID" "$dir" 2>/dev/null || true
    fi
done

if [ -x /opt/nvidia/nvidia_entrypoint.sh ]; then
    exec /opt/nvidia/nvidia_entrypoint.sh gosu "$USER_ID:$GROUP_ID" "$@"
else
    exec gosu "$USER_ID:$GROUP_ID" "$@"
fi
