from typing import Any

import cv2
import numpy as np
import SimpleITK as sitk
from skimage.morphology import remove_small_objects


def remove_small_true_false_objects(mask: np.ndarray, threshold: int = 3000) -> Any:
    new_mask = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(mask.shape[0]):
        contours, _ = cv2.findContours(
            mask[i, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(new_mask[i, :, :], contours, -1, color=1, thickness=cv2.FILLED)
    new_mask = new_mask.astype(bool)
    remove_small_objects(new_mask, min_size=threshold, connectivity=10, out=new_mask)
    np.invert(new_mask, out=new_mask)
    remove_small_objects(new_mask, min_size=threshold, connectivity=10, out=new_mask)
    np.invert(new_mask, out=new_mask)
    return new_mask


def postprocess_part_segmentation(
    trunc_img: sitk.Image,
    extr_img: sitk.Image,
):
    trunc = remove_small_true_false_objects(sitk.GetArrayViewFromImage(trunc_img))
    extr = remove_small_true_false_objects(sitk.GetArrayViewFromImage(extr_img))

    full_img = np.zeros(trunc.shape, dtype=np.uint8)
    full_img[extr] = 2
    full_img[trunc] = 1
    body_parts = sitk.GetImageFromArray(full_img)
    body_parts.CopyInformation(trunc_img)  # type: ignore
    return body_parts


# if __name__ == "__main__":
#     seg = (
#         Path().cwd().parent.parent.parent
#         / "qualitychecker"
#         / "debug"
#         / "f3959da3-23761c77-e5aba619-c92c5492-3740edf5"
#         / "segmentations"
#     )
#     postprocess_part_segmentation(seg)
