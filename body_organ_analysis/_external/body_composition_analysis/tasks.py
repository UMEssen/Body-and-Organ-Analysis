task_vals = {
    "body_parts": {
        "task_id": 543,
        "resample": 5.0,
        "folds": [0, 1, 2, 3, 4],
        "resample_only_thickness": True,
        "trainer": "nnUNetTrainer_1500epochs_NoMirroring",
        "crop": None,
    },
    "bca": {
        "task_id": 542,
        "resample": 5.0,
        "folds": [0, 1, 2, 3, 4],
        "resample_only_thickness": True,
        "trainer": "nnUNetTrainerV2",
        "crop": None,
    },
}


def get_task_info(task_name: str, fast: bool = False):
    return task_vals[task_name]
