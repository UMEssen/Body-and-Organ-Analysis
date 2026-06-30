import os
import unittest
from unittest import mock

from body_organ_analysis.compute.config import (
    env_bool,
    env_str,
    resolve_device,
    resolve_models,
)
from body_organ_analysis.compute.constants import ALL_MODELS, LICENSE_MODELS

ALL_RESOLVED = set(ALL_MODELS) - {"body_parts", "body_regions"}


class TestResolveModels(unittest.TestCase):
    def test_none_returns_all(self) -> None:
        self.assertEqual(resolve_models(None), ALL_RESOLVED)

    def test_all_keyword(self) -> None:
        self.assertEqual(resolve_models("all"), ALL_RESOLVED)
        self.assertEqual(resolve_models("ALL"), ALL_RESOLVED)
        self.assertEqual(resolve_models(""), ALL_RESOLVED)

    def test_all_excludes_bca_submodels(self) -> None:
        # bca is in "all"; its run_pipeline computes body_parts/body_regions
        # internally, so they must be dropped to avoid running them twice.
        resolved = resolve_models("all")
        self.assertIn("bca", resolved)
        self.assertIn("total", resolved)
        self.assertNotIn("body_parts", resolved)
        self.assertNotIn("body_regions", resolved)

    def test_plus_split(self) -> None:
        self.assertEqual(resolve_models("total+body_parts"), {"total", "body_parts"})

    def test_bca_implicitly_adds_total(self) -> None:
        self.assertEqual(resolve_models("bca"), {"bca", "total"})

    def test_hyphen_is_normalized_to_underscore(self) -> None:
        self.assertEqual(resolve_models("body-parts"), {"body_parts"})

    def test_invalid_dropped_when_not_strict(self) -> None:
        # The legacy "body" name is not in ALL_MODELS and is silently dropped.
        self.assertEqual(resolve_models("body+total"), {"total"})

    def test_invalid_raises_when_strict(self) -> None:
        with self.assertRaises(ValueError):
            resolve_models("body+total", strict=True)

    def test_all_excludes_heartchambers_without_license(self) -> None:
        # Without a license, heartchambers_highres is not part of "all".
        self.assertEqual(resolve_models("all"), ALL_RESOLVED)
        self.assertNotIn("heartchambers_highres", resolve_models("all"))

    def test_all_with_valid_license_adds_heartchambers(self) -> None:
        with mock.patch("totalsegmentator.config.is_valid_license", return_value=True):
            self.assertEqual(
                resolve_models("all", license_number="123456789012345678"),
                ALL_RESOLVED | LICENSE_MODELS,
            )

    def test_all_with_invalid_license_keeps_default(self) -> None:
        with mock.patch("totalsegmentator.config.is_valid_license", return_value=False):
            self.assertEqual(resolve_models("all", license_number="bad"), ALL_RESOLVED)

    def test_explicit_spec_ignores_license(self) -> None:
        # A valid license only augments "all"; explicit specs are untouched.
        with mock.patch(
            "totalsegmentator.config.is_valid_license", return_value=True
        ) as is_valid:
            self.assertEqual(
                resolve_models("total", license_number="123456789012345678"),
                {"total"},
            )
            is_valid.assert_not_called()

    def test_heartchambers_is_selectable(self) -> None:
        # It is valid when requested explicitly, in both lenient and strict mode.
        self.assertEqual(
            resolve_models("total+heartchambers_highres"),
            {"total", "heartchambers_highres"},
        )
        self.assertEqual(
            resolve_models("total+heartchambers_highres", strict=True),
            {"total", "heartchambers_highres"},
        )

    def test_bca_with_submodels(self) -> None:
        self.assertEqual(
            resolve_models("bca+body_regions+body_parts"),
            {"bca", "total"},
        )

    def test_submodels_kept_without_bca(self) -> None:
        # Without bca the submodels are run directly, so the dedup must not strip them.
        self.assertEqual(
            resolve_models("body_regions+body_parts"),
            {"body_regions", "body_parts"},
        )


class TestResolveDevice(unittest.TestCase):
    def test_default_is_gpu(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device(), "gpu")

    def test_cuda_is_aliased_to_gpu(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device("cuda"), "gpu")

    def test_cpu_passthrough(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device("cpu"), "cpu")

    def test_explicit_gpu_id_sets_visible_devices(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_device("gpu:2"), "gpu:2")
            self.assertEqual(os.environ["NVIDIA_VISIBLE_DEVICES"], "2")

    def test_nvidia_id_is_used_as_fallback(self) -> None:
        with mock.patch.dict(os.environ, {"NVIDIA_ID": "3"}, clear=True):
            self.assertEqual(resolve_device("gpu"), "gpu:3")


class TestEnvParsing(unittest.TestCase):
    def test_env_bool_true_values(self) -> None:
        for raw in ("1", "true", "TRUE", " True "):
            with mock.patch.dict(os.environ, {"BOA_X": raw}, clear=True):
                self.assertTrue(env_bool("BOA_X"))

    def test_env_bool_false_and_default(self) -> None:
        with mock.patch.dict(os.environ, {"BOA_X": "0"}, clear=True):
            self.assertFalse(env_bool("BOA_X"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(env_bool("BOA_X", True))

    def test_env_str(self) -> None:
        with mock.patch.dict(os.environ, {"BOA_Y": "  hi "}, clear=True):
            self.assertEqual(env_str("BOA_Y"), "hi")
        with mock.patch.dict(os.environ, {"BOA_Y": "TODO"}, clear=True):
            self.assertEqual(env_str("BOA_Y", "fallback"), "fallback")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(env_str("BOA_Y"))


if __name__ == "__main__":
    unittest.main()
