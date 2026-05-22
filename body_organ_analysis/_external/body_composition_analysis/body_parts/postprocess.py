import cv2
import numpy as np
import SimpleITK as sitk
from skimage.morphology import remove_small_objects


def remove_small_labeled_objects(mask: np.ndarray, threshold: int = 3000) -> np.ndarray:
    """
    Remove small connected components and small holes for every label > 0.

    Parameters
    ----------
    mask:
        3D label image with shape (z, y, x)
    threshold:
        Minimum size of connected components / holes to keep

    Returns
    -------
    np.ndarray
        Cleaned label image with the same labels as the input
    """
    out = np.zeros(mask.shape, dtype=mask.dtype)

    labels = np.unique(mask)
    labels = labels[labels > 0]

    for label in labels:
        label_mask = mask == label

        # Fill 2D contours slice-wise, same as your original logic
        filled = np.zeros(label_mask.shape, dtype=np.uint8)
        for i in range(label_mask.shape[0]):
            slice_mask = label_mask[i].astype(np.uint8)
            contours, _ = cv2.findContours(
                slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(filled[i], contours, -1, color=1, thickness=cv2.FILLED)

        filled = filled.astype(bool)

        # Remove small foreground objects
        remove_small_objects(filled, max_size=threshold - 1, connectivity=3, out=filled)

        # Remove small holes
        np.invert(filled, out=filled)
        remove_small_objects(filled, max_size=threshold - 1, connectivity=3, out=filled)
        np.invert(filled, out=filled)

        out[filled] = label

    return out


def postprocess_part_segmentation(img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.uint8, copy=False)
    arr = remove_small_labeled_objects(arr)
    new_img = sitk.GetImageFromArray(arr)
    new_img.CopyInformation(img)
    return new_img
