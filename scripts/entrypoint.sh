#!/bin/bash
set -e

DOCKER_USER="${DOCKER_USER:-1000:1000}"
USER_ID="${DOCKER_USER%:*}"
GROUP_ID="${DOCKER_USER#*:}"
HOME_DIR="${DOCKER_HOME:-/home/boa}"
XDG_CACHE_HOME_DIR="${XDG_CACHE_HOME:-$HOME_DIR/.cache}"
XDG_CONFIG_HOME_DIR="${XDG_CONFIG_HOME:-$HOME_DIR/.config}"

mkdir -p "$HOME_DIR" "$XDG_CACHE_HOME_DIR" "$XDG_CONFIG_HOME_DIR"

for dir in /storage_directory /workspace "$HOME_DIR"; do
    if [ -d "$dir" ]; then
        chown -R "$USER_ID:$GROUP_ID" "$dir" 2>/dev/null || true
    fi
done

# Fix output folder ownership and chrome installation issues
command=(
    gosu "$USER_ID:$GROUP_ID"
    env
    HOME="$HOME_DIR"
    XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"
    XDG_CONFIG_HOME="$XDG_CONFIG_HOME_DIR"
    "$@"
)

if [ -x /opt/nvidia/nvidia_entrypoint.sh ]; then
    exec /opt/nvidia/nvidia_entrypoint.sh "${command[@]}"
else
    exec "${command[@]}"
fi
