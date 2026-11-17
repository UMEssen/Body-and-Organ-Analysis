from pathlib import Path
from unittest import mock

import pytest
from pydicom.dataset import Dataset

from body_organ_analysis.compute import io
from body_organ_analysis.compute.io import get_image_info, validate_dicom


def _derived_ct() -> Dataset:
    dcm = Dataset()
    dcm.Modality = "CT"
    dcm.ImageType = ["DERIVED", "SECONDARY", "AXIAL"]
    return dcm


@pytest.mark.parametrize(
    ("env", "rejected"),
    [
        ({}, True),
        ({"IMAGE_TYPE_BAD_MARKERS": "LOCALIZER"}, False),
        ({"IMAGE_TYPE_BAD_MARKERS": "SECONDARY"}, True),
        ({"IMAGE_TYPE_BAD_MARKERS": ""}, True),
        ({"IMAGE_TYPE_BAD_MARKERS": "none"}, False),
    ],
    indirect=["env"],
)
@pytest.mark.usefixtures("env")
def test_validate_dicom_image_type_markers(rejected: bool) -> None:
    assert (validate_dicom(_derived_ct(), num_dicoms=100) is not None) is rejected


@pytest.mark.parametrize(
    ("image_type", "expected"),
    [
        (["ORIGINAL", "PRIMARY", "AXIAL"], r"ORIGINAL\PRIMARY\AXIAL"),
        ("ORIGINAL", "ORIGINAL"),
        (None, None),
    ],
)
def test_get_image_info_image_type(
    tmp_path: Path, image_type: list[str] | str | None, expected: str | None
) -> None:
    dcm = Dataset()
    dcm.Modality = "CT"
    if image_type is not None:
        dcm.ImageType = image_type
    with (
        mock.patch.object(
            io, "_load_series_from_disk", return_value=(None, ["x"] * 100)
        ),
        mock.patch.object(io.pydicom, "dcmread", return_value=dcm),
        mock.patch.object(io.sitk, "WriteImage"),
    ):
        _, ct_info = get_image_info(tmp_path, tmp_path / "out")
    assert {"name": "ImageType", "value": expected} in ct_info
