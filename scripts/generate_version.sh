#!/usr/bin/env bash

# Writes the git hash into a generated module and exports the package version and
# git hash. Source it (`source scripts/generate_version.sh`) to reuse the exported
# GIT_HASH / PACKAGE_VERSION in the calling shell.

TARGET_FILE=body_organ_analysis/_githash.py
GIT_HASH=$(git rev-parse --short HEAD)
PACKAGE_VERSION=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' body_organ_analysis/_version.py)

echo "# This file was generated automatically, do not edit" > "$TARGET_FILE"
echo "__githash__ = \"$GIT_HASH\"" >> "$TARGET_FILE"

export GIT_HASH
export PACKAGE_VERSION
