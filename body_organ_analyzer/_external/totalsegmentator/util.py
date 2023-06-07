from typing import Optional, Union, List
import numpy as np

ADDITIONAL_MODELS_OUTPUT_NAME = {
    "lung_vessels": "lung_vessels_airways",
    "cerebral_bleed": "cerebral_bleed",
    "hip_implant": "hip_implant",
    "coronary_arteries": "coronary_arteries",
    "pleural_pericard_effusion": "pleural_pericard_effusion",
    "liver_vessels": "liver_vessels",
}


def convert_resampling_slices(
    slices: int, current_sampling: float, target_resampling: Optional[float]
) -> int:
    if target_resampling is None:
        return slices
    return round((slices / target_resampling) * current_sampling)


def create_mask(region_data: np.ndarray, labels: Union[int, List[int]]) -> np.ndarray:
    mask = np.zeros(region_data.shape, dtype=bool)
    if isinstance(labels, int):
        mask[region_data == labels] = True
    else:
        mask[np.isin(region_data, labels)] = True
    return mask
