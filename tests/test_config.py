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


class TestResolveModels(unittest.TestCase):
    def test_none_returns_all(self) -> None:
        self.assertEqual(resolve_models(None), set(ALL_MODELS))

    def test_all_keyword(self) -> None:
        self.assertEqual(resolve_models("all"), set(ALL_MODELS))
        self.assertEqual(resolve_models("ALL"), set(ALL_MODELS))
        self.assertEqual(resolve_models(""), set(ALL_MODELS))

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
        self.assertEqual(resolve_models("all"), set(ALL_MODELS))
        self.assertNotIn("heartchambers_highres", resolve_models("all"))

    def test_all_with_valid_license_adds_heartchambers(self) -> None:
        with mock.patch("totalsegmentator.config.is_valid_license", return_value=True):
            self.assertEqual(
                resolve_models("all", license_number="123456789012345678"),
                set(ALL_MODELS) | LICENSE_MODELS,
            )

    def test_all_with_invalid_license_keeps_default(self) -> None:
        with mock.patch("totalsegmentator.config.is_valid_license", return_value=False):
            self.assertEqual(
                resolve_models("all", license_number="bad"), set(ALL_MODELS)
            )

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
