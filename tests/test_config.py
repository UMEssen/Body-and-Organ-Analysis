import os
from unittest import mock

import pytest

from body_organ_analysis.compute.config import (
    env_bool,
    env_str,
    resolve_device,
    resolve_models,
)
from body_organ_analysis.compute.constants import ALL_MODELS, LICENSE_MODELS

# bca computes body_parts/body_regions internally, so "all" drops them to avoid
# running them twice. Without a license, heartchambers_highres is not included.
ALL_RESOLVED = set(ALL_MODELS) - {"body_parts", "body_regions"}
LICENSE = "123456789012345678"


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (None, ALL_RESOLVED),
        ("all", ALL_RESOLVED),
        ("ALL", ALL_RESOLVED),
        ("", ALL_RESOLVED),
        ("total+body_parts", {"total", "body_parts"}),
        ("bca", {"bca", "total"}),
        ("bca+body_regions+body_parts", {"bca", "total"}),
        # Without bca the submodels are run directly and must be kept
        ("body_regions+body_parts", {"body_regions", "body_parts"}),
        ("body-parts", {"body_parts"}),
        # The legacy "body" name is not in ALL_MODELS and is silently dropped
        ("body+total", {"total"}),
        ("total+heartchambers_highres", {"total", "heartchambers_highres"}),
    ],
)
def test_resolve_models(spec: str | None, expected: set[str]) -> None:
    assert resolve_models(spec) == expected


def test_resolve_models_strict_accepts_heartchambers() -> None:
    assert resolve_models("total+heartchambers_highres", strict=True) == {
        "total",
        "heartchambers_highres",
    }


def test_resolve_models_strict_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        resolve_models("body+total", strict=True)


@pytest.mark.parametrize(
    ("spec", "valid", "expected"),
    [
        ("all", True, ALL_RESOLVED | LICENSE_MODELS),
        ("all", False, ALL_RESOLVED),
        # A valid license only augments "all"; explicit specs are untouched
        ("total", True, {"total"}),
    ],
)
def test_resolve_models_with_license(
    spec: str, valid: bool, expected: set[str]
) -> None:
    with mock.patch("totalsegmentator.config.is_valid_license", return_value=valid):
        assert resolve_models(spec, license_number=LICENSE) == expected


def test_explicit_spec_skips_license_check() -> None:
    with mock.patch("totalsegmentator.config.is_valid_license") as is_valid:
        resolve_models("total", license_number=LICENSE)
    is_valid.assert_not_called()


@pytest.mark.parametrize(
    ("env", "device", "expected"),
    [
        ({}, None, "gpu"),
        ({}, "cuda", "gpu"),
        ({}, "cpu", "cpu"),
        ({}, "gpu:2", "gpu:2"),
        ({"NVIDIA_ID": "3"}, "gpu", "gpu:3"),
    ],
    indirect=["env"],
)
@pytest.mark.usefixtures("env")
def test_resolve_device(device: str | None, expected: str) -> None:
    assert resolve_device(device) == expected


@pytest.mark.usefixtures("env")
def test_explicit_gpu_id_sets_visible_devices() -> None:
    resolve_device("gpu:2")
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "2"


@pytest.mark.parametrize(
    ("env", "default", "expected"),
    [
        ({"BOA_X": "1"}, False, True),
        ({"BOA_X": "true"}, False, True),
        ({"BOA_X": "TRUE"}, False, True),
        ({"BOA_X": " True "}, False, True),
        ({"BOA_X": "0"}, True, False),
        ({}, True, True),
    ],
    indirect=["env"],
)
@pytest.mark.usefixtures("env")
def test_env_bool(default: bool, expected: bool) -> None:
    assert env_bool("BOA_X", default) is expected


@pytest.mark.parametrize(
    ("env", "default", "expected"),
    [
        ({"BOA_Y": "  hi "}, None, "hi"),
        ({"BOA_Y": "TODO"}, "fallback", "fallback"),
        ({}, None, None),
    ],
    indirect=["env"],
)
@pytest.mark.usefixtures("env")
def test_env_str(default: str | None, expected: str | None) -> None:
    assert env_str("BOA_Y", default) == expected
