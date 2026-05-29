SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

SH_FILES := $(wildcard scripts/*.sh example_scripts/*.sh)
COMPOSE_FILES := docker-compose.yml docker-compose-win.yml
DOCKERFILES := $(wildcard scripts/*.dockerfile)

.PHONY: help sh-check compose-check docker-check pre-commit test check all

help:
	@echo "Targets:"
	@echo "  sh-check       bash -n on $(words $(SH_FILES)) shell scripts"
	@echo "  compose-check  docker compose config on $(COMPOSE_FILES)"
	@echo "  docker-check   docker buildx build --check on $(words $(DOCKERFILES)) Dockerfiles (buildx >= 0.15)"
	@echo "  pre-commit     pre-commit run --all-files"
	@echo "  test           pytest"
	@echo "  check          sh-check + compose-check + docker-check"
	@echo "  all            check + pre-commit + test"

sh-check:
	@for f in $(SH_FILES); do echo "bash -n $$f"; bash -n "$$f"; done

compose-check:
	@# --no-interpolate validates structure without a populated env: unset vars
	@# would otherwise collapse volume specs like ${POSTGRES_DATA}:/path into an
	@# invalid ":/path" ("empty section between colons") and spam warnings.
	@for f in $(COMPOSE_FILES); do echo "docker compose -f $$f config"; docker compose -f "$$f" config -q --no-interpolate; done

docker-check:
	@for f in $(DOCKERFILES); do echo "docker buildx build --check -f $$f ."; docker buildx build --check -f "$$f" .; done

pre-commit:
	pre-commit run --all-files

test:
	pytest

check: sh-check compose-check docker-check

all: check pre-commit test
