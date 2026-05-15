from typing import TypedDict

import nibabel


class TaskConfig(TypedDict):
    task_id: int
    resample: float
    folds: list[int]
    resample_only_thickness: bool
    trainer: str
    crop: nibabel.Nifti1Image | None


task_vals: dict[str, TaskConfig] = {
    "body_parts": {
        "task_id": 543,
        "resample": 5.0,
        "folds": [0, 1, 2, 3, 4],
        "resample_only_thickness": True,
        "trainer": "nnUNetTrainer_1500epochs_NoMirroring",
        "crop": None,
    },
    "body_regions": {
        "task_id": 542,
        "resample": 5.0,
        "folds": [0, 1, 2, 3, 4],
        "resample_only_thickness": True,
        "trainer": "nnUNetTrainerNoMirroring",
        "crop": None,
    },
}


def get_task_info(task_name: str, _fast: bool = False) -> TaskConfig:
    return task_vals[task_name]
