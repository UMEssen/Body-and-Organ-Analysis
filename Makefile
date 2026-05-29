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
	@# Validate against the committed sample env so the *interpolated* output is
	@# checked (and .env_sample is kept in sync with the compose files). With an
	@# empty env, unset vars would collapse volume specs like ${POSTGRES_DATA}:/p
	@# into an invalid ":/p", and a literal ${ORTHANC_PORT} in a ports field is
	@# rejected as "invalid hostPort" by stricter compose versions.
	@for f in $(COMPOSE_FILES); do echo "docker compose -f $$f config"; docker compose -f "$$f" --env-file .env_sample config -q; done

docker-check:
	@for f in $(DOCKERFILES); do echo "docker buildx build --check -f $$f ."; docker buildx build --check -f "$$f" .; done

pre-commit:
	pre-commit run --all-files

test:
	pytest

check: sh-check compose-check docker-check

all: check pre-commit test
