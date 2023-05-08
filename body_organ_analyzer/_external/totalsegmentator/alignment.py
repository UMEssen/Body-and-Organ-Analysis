import sys

import numpy as np
import nibabel as nib
import nibabel.processing
import logging

logger = logging.getLogger(__name__)


def as_closest_canonical(img_in):
    """
    Convert the given nifti file to the closest canonical nifti file.
    """
    return nib.as_closest_canonical(img_in)


def as_closest_canonical_nifti(path_in, path_out):
    """
    Convert the given nifti file to the closest canonical nifti file.
    """
    img_in = nib.load(path_in)
    img_out = nib.as_closest_canonical(img_in)
    nib.save(img_out, path_out)


def undo_canonical(img_can, img_orig):
    """
    Inverts nib.to_closest_canonical()

    img_can: the image we want to move back
    img_orig: the original image because transforming to canonical

    returns image in original space

    https://github.com/nipy/nibabel/issues/1063
    """
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

    img_ornt = io_orientation(img_orig.affine)
    ras_ornt = axcodes2ornt("RAS")

    to_canonical = img_ornt  # Same as ornt_transform(img_ornt, ras_ornt)
    from_canonical = ornt_transform(ras_ornt, img_ornt)

    # Same as as_closest_canonical
    # img_canonical = img_orig.as_reoriented(to_canonical)

    return img_can.as_reoriented(from_canonical)


def undo_canonical_nifti(path_in_can, path_in_orig, path_out):
    e = nib.load(path_in_can)
    img_orig = nib.load(path_in_orig)
    img_out = undo_canonical(img_can, img_orig)
    nib.save(img_out, path_out)


def load_nibabel_image_with_axcodes(image, axcodes="RAS"):
    input_axcodes = "".join(nibabel.aff2axcodes(image.affine))
    axcodes = "".join(axcodes)
    if input_axcodes != axcodes:
        logger.debug(
            f"Input image does not match the required axcodes: got {input_axcodes}, "
            f"expected {axcodes}. Automatically reorienting the image volume."
        )
        input_ornt = nibabel.orientations.axcodes2ornt(input_axcodes)
        expected_ornt = nibabel.orientations.axcodes2ornt(axcodes)
        transform = nibabel.orientations.ornt_transform(input_ornt, expected_ornt)
        return image.as_reoriented(transform)
    return image


def convert_nibabel_to_orginal_with_axcodes(
    image_transformed, image_original, transformed_axcodes="RAS"
):
    img_ornt = nibabel.orientations.io_orientation(image_original.affine)
    ras_ornt = nibabel.orientations.axcodes2ornt(transformed_axcodes)
    from_canonical = nibabel.orientations.ornt_transform(ras_ornt, img_ornt)
    return image_transformed.as_reoriented(from_canonical)
