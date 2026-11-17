import pytest
from pydicom.dataset import Dataset

from body_organ_analysis.compute.io import validate_dicom


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
