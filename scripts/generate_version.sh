#!/usr/bin/env bash

TARGET_FILE=body_organ_analysis/_githash.py
GIT_VERSION=$(git rev-parse --short HEAD)

echo "# This file was generated automatically, do not edit" > "$TARGET_FILE"
echo "__githash__ = \"$GIT_VERSION\"" >> "$TARGET_FILE"
