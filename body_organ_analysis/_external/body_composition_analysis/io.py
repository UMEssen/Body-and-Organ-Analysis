import logging
import pathlib

import nibabel
import numpy as np
import SimpleITK as sitk
from totalsegmentator.alignment import load_nibabel_image_with_axcodes

logger = logging.getLogger(__name__)


def nib_to_sitk(image: nibabel.spatialimages.SpatialImage) -> sitk.Image:
    # From https://github.com/Kimerth/torchio/blob/19639037a530d31e8ba487e0945152ca765b8b8a/transforms/transform.py#L50-L62
    FLIP_XY = np.diag((-1, -1, 1))
    data = np.asanyarray(image.dataobj)
    affine = image.affine
    origin = np.dot(FLIP_XY, affine[:3, 3]).astype(np.float64)
    RZS = affine[:3, :3]
    spacing = np.sqrt(np.sum(RZS * RZS, axis=0))
    R = RZS / spacing
    direction = np.dot(FLIP_XY, R).flatten()
    image = sitk.GetImageFromArray(data.transpose())
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image


def make_affine(image: sitk.Image) -> np.ndarray:
    # get affine transform in LPS
    c = [
        image.TransformContinuousIndexToPhysicalPoint(p)
        for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))
    ]
    c = np.array(c)
    affine = np.concatenate(
        [np.concatenate([c[0:3] - c[3:], c[3:]], axis=0), [[0.0], [0.0], [0.0], [1.0]]],
        axis=1,
    )
    affine = np.transpose(affine)
    # convert to RAS to match nibabel
    affine = np.matmul(np.diag([-1.0, -1.0, 1.0, 1.0]), affine)
    return affine


def sitk_to_nib(image: sitk.Image) -> nibabel.spatialimages.SpatialImage:
    # https://niftynet.readthedocs.io/en/v0.2.1/_modules/niftynet/io/simple_itk_as_nibabel.html
    assert isinstance(image, sitk.Image)
    return nibabel.Nifti1Image(
        sitk.GetArrayFromImage(image).transpose(), make_affine(image)
    )


def resample_to_thickness(image: sitk.Image, thickness: float) -> sitk.Image:
    input_size = image.GetSize()
    input_spacing = image.GetSpacing()
    output_spacing = (input_spacing[0], input_spacing[1], thickness)
    output_direction = image.GetDirection()
    output_origin = image.GetOrigin()
    output_size = [
        int(np.round(input_size[i] * input_spacing[i] / output_spacing[i]))
        for i in range(3)
    ]
    image = sitk.Resample(
        image,
        output_size,
        sitk.Transform(),
        sitk.sitkLinear,
        output_origin,
        output_spacing,
        output_direction,
    )
    return image


def process_image(
    img: nibabel.Nifti1Image,
    resample_thickness: float = None,
) -> sitk.Image:
    image = nib_to_sitk(load_nibabel_image_with_axcodes(img, axcodes="LPS"))
    slice_thickness = image.GetSpacing()[2]
    if resample_thickness is not None and not np.isclose(
        slice_thickness, resample_thickness
    ):
        # logger.warning(
        #     f"Unexpected slice thickness found in input image: got "
        #     f"{slice_thickness:.2f}mm, expected 5.00mm. Resampling to expected slice "
        #     "thickness"
        # )
        image = resample_to_thickness(image, resample_thickness)

    return image


def load_image(path: pathlib.Path, resample_thickness: float = None) -> sitk.Image:
    return process_image(
        nibabel.load(path),
        resample_thickness=resample_thickness,
    )
