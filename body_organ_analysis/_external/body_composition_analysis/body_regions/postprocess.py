import numpy as np
import SimpleITK as sitk
import skimage.measure
from body_composition_analysis.body_regions.definition import BodyRegion


def _filter_largest_unique_segment(segmentation: np.ndarray, mask: np.ndarray) -> None:
    labels = skimage.measure.label(mask)
    props = skimage.measure.regionprops(labels)
    if len(props) > 1:
        # HACK: Cast required for type hinting
        props = sorted(props, key=lambda x: int(x.area), reverse=True)
        for prop in props[1:]:
            segmentation[np.equal(labels, prop.label)] = 255


def postprocess_region_segmentation(body_regions: sitk.Image) -> sitk.Image:
    # Extract numeric data from SimpleITK images
    seg_data = sitk.GetArrayFromImage(body_regions)

    # Some regions can only have one single segment and thus only the largest
    # should be used for further computations
    _filter_largest_unique_segment(seg_data, np.greater(seg_data, 0))
    _filter_largest_unique_segment(
        seg_data,
        np.logical_or(
            np.equal(seg_data, BodyRegion.THORACIC_CAVITY.value),
            np.logical_or(
                np.equal(seg_data, BodyRegion.MEDIASTINUM.value),
                np.equal(seg_data, BodyRegion.PERICARDIUM.value),
            ),
        ),
    )
    for region in (BodyRegion.PERICARDIUM, BodyRegion.ABDOMINAL_CAVITY):
        _filter_largest_unique_segment(seg_data, np.equal(seg_data, region.value))

    result = sitk.GetImageFromArray(seg_data)
    result.CopyInformation(body_regions)
    return result
