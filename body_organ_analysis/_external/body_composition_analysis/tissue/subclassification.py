import pathlib

import numpy as np
import scipy.ndimage
import SimpleITK as sitk
from body_composition_analysis.tissue.definition import TISSUE_DERIVATION_RULES, HURange


def subclassify_tissues(
    image: sitk.Image,
    body_regions: sitk.Image,
    output_dir: pathlib.Path,
    median_filtering: bool = False,
    orientation: str = None,
) -> sitk.Image:
    image_data = sitk.GetArrayFromImage(image)
    seg_data = sitk.GetArrayFromImage(body_regions)
    # Reduce the influence of noise on subclassification by applying a median
    # filter on the CT image. However, due to the large slice thickness it
    # should be better to only filter in-plane with a 2D kernel.
    if median_filtering:
        # TODO: Do this better, maybe with the body_regions.GetDirections()?
        assert orientation is not None
        if "I" in orientation:
            slice_position = orientation[::-1].index("I")
        elif "S" in orientation:
            slice_position = orientation[::-1].index("S")
        else:
            raise ValueError(
                f"The orientation {orientation} does not contain neither I nor S."
            )
        kernel = [3, 3, 3]
        kernel[slice_position] = 1
        image_data = scipy.ndimage.median_filter(image_data, size=kernel)

    # Tissue subclassification requires hounsfield unit masks and they should
    # only be computed once
    precomputed_masks = {}
    for hu_range in HURange:
        precomputed_masks[hu_range] = np.logical_and(
            np.greater_equal(image_data, hu_range.value[0]),
            np.less_equal(image_data, hu_range.value[1]),
        )

    # Derive each tissue from the semantic body region and a hounsfield unit
    # range according to the derivation table
    tissue_data = np.zeros_like(seg_data)
    for tissue, (hu_mask, region) in TISSUE_DERIVATION_RULES.items():
        region_mask = np.equal(seg_data, region.value)
        tissue_mask = precomputed_masks[hu_mask] & region_mask
        tissue_data[tissue_mask] = tissue.value

    if ".nii" not in output_dir.name:
        output_path = output_dir / "tissues.nii.gz"
    else:
        output_path = output_dir

    result = sitk.GetImageFromArray(tissue_data)
    result.CopyInformation(body_regions)
    sitk.WriteImage(result, str(output_path))
    return result
