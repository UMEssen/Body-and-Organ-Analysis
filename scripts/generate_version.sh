#!/usr/bin/env bash

TARGET_FILE=body_organ_analyzer/_version.py
export PACKAGE_VERSION=$(cat pyproject.toml | grep -e "^version" | grep -o -e "[0-9]\+\.[0-9]\+\.[0-9]\+")
export GIT_VERSION=$(git rev-parse --short HEAD)

echo "# This file was generated automatically, do not edit" > $TARGET_FILE
echo "__version__ = \"$PACKAGE_VERSION\"" >> $TARGET_FILE
echo "__githash__ = \"$GIT_VERSION\"" >> $TARGET_FILE
