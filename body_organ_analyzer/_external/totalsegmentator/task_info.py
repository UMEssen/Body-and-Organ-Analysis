import warnings
import logging

logger = logging.getLogger(__name__)
task_vals = {
    "total-fast": {
        "task_id": 256,
        "resample": 3.0,
        "trainer": "nnUNetTrainerV2_ep8000_nomirror",
        "crop": None,
    },
    "total": {
        "task_id": [251, 252, 253, 254, 255],
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2_ep4000_nomirror",
        "crop": None,
    },
    "organs": {
        "task_id": [251],
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2_ep4000_nomirror",
        "crop": None,
    },
    "vertebrae": {
        "task_id": [252],
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2_ep4000_nomirror",
        "crop": None,
    },
    "cardiac": {
        "task_id": [253],
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2_ep4000_nomirror",
        "crop": None,
    },
    "muscles": {
        "task_id": [254],
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2_ep4000_nomirror",
        "crop": None,
    },
    "ribs": {
        "task_id": [255],
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2_ep4000_nomirror",
        "crop": None,
    },
    "lung_vessels": {
        "task_id": 258,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "lung",
    },
    "covid": {
        "task_id": 201,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "lung",
    },
    "cerebral_bleed": {
        "task_id": 150,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "brain",
    },
    "hip_implant": {
        "task_id": 260,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "pelvis",
    },
    "coronary_arteries": {
        "task_id": 503,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "heart",
    },
    "body-fast": {
        "task_id": 269,
        "resample": 6.0,
        "trainer": "nnUNetTrainerV2",
        "crop": None,
    },
    "body": {
        "task_id": 273,
        "resample": 1.5,
        "trainer": "nnUNetTrainerV2",
        "crop": None,
    },
    "pleural_pericard_effusion": {
        "task_id": 315,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "lung",
    },
    "liver_vessels": {
        "task_id": 8,
        "resample": None,
        "trainer": "nnUNetTrainerV2",
        "crop": "liver",
    },
    "bca": {
        "task_id": 542,
        "resample": 5.0,
        "resample_only_thickness": True,
        "trainer": "nnUNetTrainerV2",
        "crop": None,
    },
}


def get_task_info(
    task_name: str,
    fast: bool = False,
    multilabel_image: bool = False,
    quiet: bool = False,
):
    if fast and task_name in {
        "lung_vessels",
        "covid",
        "cerebral_bleed",
        "hip_implant",
        "coronary_arteries",
        "pleural_pericard_effusion",
        "liver_vessels",
        "bca",
    }:
        warnings.warn(
            f"task {task_name} does not work with option --fast, setting fast to false."
        )
        fast = False
    if multilabel_image and task_name in {"lung_vessels", "body"}:
        warnings.warn(
            f"task {task_name} does not work with one multilabel image, "
            f"because of postprocessing, setting multilabel_image to false."
        )
        multilabel_image = False

    complete_task_name = task_name + ("-fast" if fast else "")
    if complete_task_name in {"total-fast", "body-fast"} and not quiet:
        logger.info("Using 'fast' option: resampling to lower resolution (3mm)")

    # Default values
    folds = [0]
    crop_addon = [3, 3, 3]
    model = "3d_fullres"

    if task_name == "covid":
        logger.warning(
            "WARNING: The COVID model finds many types of lung opacity not only COVID. Use with care!"
        )
    if task_name == "coronary_arteries":
        logger.warning(
            "WARNING: The coronary artery model does not work very robustly. Use with care!"
        )
    if task_name == "pleural_pericard_effusion":
        crop_addon = [50, 50, 50]
        # folds = None # TODO Not used
    elif task_name == "liver_vessels":
        crop_addon = [20, 20, 20]
        # folds = None # TODO Not used
    elif task_name == "bca":
        folds = [0, 1, 2, 3, 4]

    return {
        **task_vals[complete_task_name],
        "model": model,
        "folds": folds,
        "multilabel_image": multilabel_image,
        "crop_addon": crop_addon,
    }
