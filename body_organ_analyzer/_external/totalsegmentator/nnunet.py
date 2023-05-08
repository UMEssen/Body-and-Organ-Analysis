import os
import time
import shutil
import subprocess
from pathlib import Path
from os.path import join
import numpy as np
import nibabel as nib
from functools import partial
from p_tqdm import p_map
import tempfile

from totalsegmentator.libs import nostdout
from scipy.ndimage import binary_dilation
import logging

logger = logging.getLogger(__name__)

with nostdout():
    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import (
        default_plans_identifier,
        network_training_output_dir,
        default_trainer,
    )

from totalsegmentator.map_to_binary import (
    class_map,
    class_map_5_parts,
    map_taskid_to_partname,
)
from totalsegmentator.alignment import (
    load_nibabel_image_with_axcodes,
    convert_nibabel_to_orginal_with_axcodes,
)
from totalsegmentator.resampling import change_spacing
from totalsegmentator.preview import generate_preview
from totalsegmentator.libs import (
    combine_masks,
    compress_nifti,
    check_if_shape_and_affine_identical,
)
from totalsegmentator.cropping import crop_to_mask_nifti, undo_crop_nifti
from totalsegmentator.cropping import crop_to_mask, undo_crop
from totalsegmentator.postprocessing import remove_outside_of_mask, extract_skin
from totalsegmentator.nifti_ext_header import save_multilabel_nifti


def _get_full_task_name(task_id: int, src: str = "raw"):
    if src == "raw":
        base = Path(os.environ["nnUNet_raw_data_base"]) / "nnUNet_raw_data"
    elif src == "preprocessed":
        base = Path(os.environ["nnUNet_preprocessed"])
    elif src == "results":
        base = Path(os.environ["RESULTS_FOLDER"]) / "nnUNet" / "3d_fullres"
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir

    # If not found in 3d_fullres, search in 2d
    if src == "results":
        base = Path(os.environ["RESULTS_FOLDER"]) / "nnUNet" / "2d"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    raise ValueError(f"task_id {task_id} not found")


def contains_empty_img(imgs):
    """
    imgs: List of image pathes
    """
    is_empty = True
    for img in imgs:
        this_is_empty = len(np.unique(nib.load(img).get_fdata())) == 1
        is_empty = is_empty and this_is_empty
    return is_empty


def nnUNet_predict(
    dir_in,
    dir_out,
    task_id,
    model="3d_fullres",
    folds=None,
    trainer="nnUNetTrainerV2",
    tta=False,
):
    """
    Identical to bash function nnUNet_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    save_npz = False
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    # num_threads_preprocessing = 1
    # num_threads_nifti_save = 1
    lowres_segmentations = None
    part_id = 0
    num_parts = 1
    disable_tta = not tta
    overwrite_existing = False
    mode = "normal" if model == "2d" else "fastest"
    all_in_gpu = True
    step_size = 0.5
    chk = "model_final_checkpoint"
    disable_mixed_precision = False

    task_id = int(task_id)
    task_name = _get_full_task_name(task_id, src="results")

    # trainer_class_name = default_trainer
    # trainer = trainer_class_name
    plans_identifier = default_plans_identifier

    model_folder_name = join(
        network_training_output_dir, model, task_name, trainer + "__" + plans_identifier
    )
    print(f"using model stored in {model_folder_name}")

    predict_from_folder(
        model_folder_name,
        dir_in,
        dir_out,
        folds,
        save_npz,
        num_threads_preprocessing,
        num_threads_nifti_save,
        lowres_segmentations,
        part_id,
        num_parts,
        not disable_tta,
        overwrite_existing=overwrite_existing,
        mode=mode,
        overwrite_all_in_gpu=all_in_gpu,
        mixed_precision=not disable_mixed_precision,
        step_size=step_size,
        checkpoint_name=chk,
    )


def save_segmentation_nifti(
    class_map_item,
    tmp_dir=None,
    file_out=None,
    nora_tag=None,
    header=None,
    task_name=None,
    quiet=None,
):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    if task_name != "total" and not quiet:
        logger.info(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(
        nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path
    )
    if nora_tag != "None":
        subprocess.call(
            f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask",
            shell=True,
        )


def nnUNet_predict_image(
    file_in,
    file_out,
    task_id,
    model="3d_fullres",
    folds=None,
    trainer="nnUNetTrainerV2",
    tta=False,
    multilabel_image=True,
    resample=None,
    resample_only_thickness=False,
    crop=None,
    crop_path=None,
    task_name="total",
    nora_tag="None",
    preview=False,
    save_binary=False,
    nr_threads_resampling=1,
    nr_threads_saving=6,
    force_split=False,
    crop_addon=[3, 3, 3],
    roi_subset=None,
    axcodes="RAS",
    quiet=False,
    verbose=False,
    test=0,
):
    """
    crop: string or a nibabel image
    resample: None or float  (target spacing for all dimensions)
    """
    file_in = Path(file_in)
    if file_out is not None:
        file_out = Path(file_out)
    multimodel = type(task_id) is list

    # for debugging
    # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    # (tmp_dir).mkdir(exist_ok=True)
    # with tmp_dir as tmp_folder:
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose:
            logger.info(f"tmp_dir: {tmp_dir}")

        img_in_orig = nib.load(file_in)
        img_in = nib.Nifti1Image(
            img_in_orig.get_fdata(), img_in_orig.affine
        )  # copy img_in_orig

        if crop is not None:
            if type(crop) is str and crop in {"lung", "pelvis", "heart"}:
                combine_masks(crop_path, crop_path / f"{crop}.nii.gz", crop)
            if type(crop) is str:
                crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz").get_fdata()
            elif type(crop) is nib.Nifti1Image:
                crop_mask_img = crop.get_fdata()
            else:
                crop_mask_img = crop
            img_in, bbox = crop_to_mask(
                img_in, crop_mask_img, addon=crop_addon, dtype=np.int32, verbose=verbose
            )
            if not quiet:
                logger.info(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

        img_in = load_nibabel_image_with_axcodes(img_in, axcodes)
        if resample_only_thickness:
            resample_target = list(img_in.header.get_zooms()[:2]) + [resample]
        else:
            resample_target = [resample, resample, resample]
        if resample is not None:
            if not quiet:
                logger.info(f"Resampling...")
            st = time.time()
            img_in_shape = img_in.shape
            img_in_rsp = change_spacing(
                img_in,
                resample_target,
                order=3,
                dtype=np.int32,
                nr_cpus=nr_threads_resampling,
            )  # 4 cpus instead of 1 makes it a bit slower
            if verbose:
                logger.info(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
            if not quiet:
                logger.info(
                    f"The CT was resampled from {img_in.header.get_zooms()} "
                    f"to {tuple(resample_target)} in {time.time() - st:.2f}s. "
                    f"New size: {img_in_rsp.shape}"
                )
        else:
            img_in_rsp = img_in

        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

        # nr_voxels_thr = 512*512*900
        nr_voxels_thr = 256 * 256 * 900
        img_parts = ["s01"]
        ss = img_in_rsp.shape
        # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
        # splitting along it does not really make sense.
        do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
        # Check this again otherwise there are errors
        if force_split and ss[2] > 200:
            do_triple_split = True
        if do_triple_split:
            if not quiet:
                logger.info(f"Splitting into subparts...")
            img_parts = ["s01", "s02", "s03"]
            third = img_in_rsp.shape[2] // 3
            margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
            img_in_rsp_data = img_in_rsp.get_fdata()
            nib.save(
                nib.Nifti1Image(
                    img_in_rsp_data[:, :, : third + margin], img_in_rsp.affine
                ),
                tmp_dir / "s01_0000.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(
                    img_in_rsp_data[:, :, third + 1 - margin : third * 2 + margin],
                    img_in_rsp.affine,
                ),
                tmp_dir / "s02_0000.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(
                    img_in_rsp_data[:, :, third * 2 + 1 - margin :], img_in_rsp.affine
                ),
                tmp_dir / "s03_0000.nii.gz",
            )

        st = time.time()
        if multimodel:  # if running multiple models
            if test == 0:
                class_map_inv = {v: k for k, v in class_map[task_name].items()}
                (tmp_dir / "parts").mkdir(exist_ok=True)
                seg_combined = {}
                # iterate over subparts of image
                for img_part in img_parts:
                    img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                    seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                # Run several tasks and combine results into one segmentation
                for idx, tid in enumerate(task_id):
                    logger.info(f"Predicting part {idx} of 5 ...")
                    with nostdout(verbose):
                        nnUNet_predict(
                            tmp_dir, tmp_dir, tid, model, folds, trainer, tta
                        )
                    # iterate over models (different sets of classes)
                    for img_part in img_parts:
                        (tmp_dir / f"{img_part}.nii.gz").rename(
                            tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz"
                        )
                        seg = nib.load(
                            tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz"
                        ).get_fdata()
                        for jdx, class_name in class_map_5_parts[
                            map_taskid_to_partname[tid]
                        ].items():
                            seg_combined[img_part][seg == jdx] = class_map_inv[
                                class_name
                            ]
                # iterate over subparts of image
                for img_part in img_parts:
                    nib.save(
                        nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine),
                        tmp_dir / f"{img_part}.nii.gz",
                    )
            elif test == 1:
                logger.warning(
                    "WARNING: Using reference seg instead of prediction for testing."
                )
                shutil.copy(
                    Path("tests") / "reference_files" / "example_seg.nii.gz",
                    tmp_dir / f"s01.nii.gz",
                )
        else:
            if not quiet:
                logger.info(f"Predicting...")
            if test == 0 or test == 2:
                with nostdout(verbose):
                    nnUNet_predict(
                        tmp_dir, tmp_dir, task_id, model, folds, trainer, tta
                    )
            # elif test == 2:
            #     logger.info("WARNING: Using reference seg instead of prediction for testing.")
            #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
            elif test == 3:
                logger.warning(
                    "WARNING: Using reference seg instead of prediction for testing."
                )
                shutil.copy(
                    Path("tests")
                    / "reference_files"
                    / "example_seg_lung_vessels.nii.gz",
                    tmp_dir / f"s01.nii.gz",
                )
        if not quiet:
            logger.info("  Predicted in {:.2f}s".format(time.time() - st))

        # Combine image subparts back to one image
        if do_triple_split:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            combined_img[:, :, :third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[
                :, :, :-margin
            ]
            combined_img[:, :, third : third * 2] = nib.load(
                tmp_dir / "s02.nii.gz"
            ).get_fdata()[:, :, margin - 1 : -margin]
            combined_img[:, :, third * 2 :] = nib.load(
                tmp_dir / "s03.nii.gz"
            ).get_fdata()[:, :, margin - 1 :]
            nib.save(
                nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz"
            )

        img_pred = nib.load(tmp_dir / "s01.nii.gz")
        if preview:
            # Generate preview before upsampling so it is faster and still in canonical space
            # for better orientation.
            if not quiet:
                logger.info("Generating preview...")
            st = time.time()
            smoothing = 20
            preview_dir = file_out.parent if multilabel_image else file_out
            generate_preview(
                img_in_rsp,
                preview_dir / f"preview_{task_name}.png",
                img_pred.get_fdata(),
                smoothing,
                task_name,
            )
            if not quiet:
                logger.info("  Generated in {:.2f}s".format(time.time() - st))

        if resample is not None:
            if not quiet:
                logger.info("Resampling...")
            if verbose:
                logger.info(f"  back to original shape: {img_in_shape}")
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = change_spacing(
                img_pred,
                resample_target,
                img_in_shape,
                order=0,
                dtype=np.uint8,
                nr_cpus=nr_threads_resampling,
                force_affine=img_in.affine,
            )

        if verbose:
            logger.info("Undoing canonical...")
        img_pred = convert_nibabel_to_orginal_with_axcodes(
            image_transformed=img_pred,
            image_original=img_in_orig,
            transformed_axcodes=axcodes,
        )

        if crop is not None:
            if verbose:
                logger.info("Undoing cropping...")
            img_pred = undo_crop(img_pred, img_in_orig, bbox)

        check_if_shape_and_affine_identical(img_in_orig, img_pred)

        img_data = img_pred.get_fdata().astype(np.uint8)
        if save_binary:
            img_data = (img_data > 0).astype(np.uint8)

        if file_out is not None:
            if not quiet:
                logger.info("Saving segmentations...")
            # Copy header to make output header exactly the same as input. But change dtype otherwise it will be
            # float or int and therefore the masks will need a lot more space.
            # (infos on header: https://nipy.org/nibabel/nifti_images.html)
            new_header = img_in_orig.header.copy()
            new_header.set_data_dtype(np.uint8)

            st = time.time()
            if multilabel_image:
                file_out.parent.mkdir(exist_ok=True, parents=True)
            else:
                file_out.mkdir(exist_ok=True, parents=True)
            if multilabel_image:
                # nib.save(nib.Nifti1Image(img_data, img_pred.affine, new_header), file_out)  # recreate nifti image to ensure uint8 dtype
                img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
                save_multilabel_nifti(img_out, file_out, class_map[task_name])
                if nora_tag != "None":
                    subprocess.call(
                        f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas",
                        shell=True,
                    )
            else:  # save each class as a separate binary image
                file_out.mkdir(exist_ok=True, parents=True)

                # Select subset of classes if required
                selected_classes = class_map[task_name]
                if roi_subset is not None:
                    selected_classes = {
                        k: v for k, v in selected_classes.items() if v in roi_subset
                    }

                # Code for single threaded execution  (runtime:24s)
                if nr_threads_saving == 1:
                    for k, v in selected_classes.items():
                        binary_img = img_data == k
                        output_path = str(file_out / f"{v}.nii.gz")
                        nib.save(
                            nib.Nifti1Image(
                                binary_img.astype(np.uint8), img_pred.affine, new_header
                            ),
                            output_path,
                        )
                        if nora_tag != "None":
                            subprocess.call(
                                f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask",
                                shell=True,
                            )
                else:
                    # Code for multithreaded execution
                    #   Speed with different number of threads:
                    #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                    nib.save(img_pred, tmp_dir / "s01.nii.gz")
                    _ = p_map(
                        partial(
                            save_segmentation_nifti,
                            tmp_dir=tmp_dir,
                            file_out=file_out,
                            nora_tag=nora_tag,
                            header=new_header,
                            task_name=task_name,
                            quiet=quiet,
                        ),
                        selected_classes.items(),
                        num_cpus=nr_threads_saving,
                        disable=quiet,
                    )

                    # Multihreaded saving with same functions as in nnUNet -> same speed as p_map
                    # pool = Pool(nr_threads_saving)
                    # results = []
                    # for k, v in selected_classes.items():
                    #     results.append(pool.starmap_async(save_segmentation_nifti, ((k, v, tmp_dir, file_out, nora_tag),) ))
                    # _ = [i.get() for i in results]  # this actually starts the execution of the async functions
                    # pool.close()
                    # pool.join()
            if not quiet:
                logger.info(f"  Saved in {time.time() - st:.2f}s")

            # Postprocessing
            if task_name == "lung_vessels":
                if type(crop) is str:
                    remove_outside_of_mask(
                        file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz"
                    )
                else:
                    seg_img = nib.load(file_out / "lung_vessels.nii.gz")
                    seg = seg_img.get_fdata()
                    mask = binary_dilation(crop, iterations=1)
                    seg[mask == 0] = 0
                    nib.save(
                        nib.Nifti1Image(seg.astype(np.uint8), seg_img.affine),
                        file_out / "lung_vessels.nii.gz",
                    )

            if task_name == "body":
                if not quiet:
                    logger.info("Creating body.nii.gz")
                combine_masks(file_out, file_out / "body.nii.gz", "body")
                if not quiet:
                    logger.info("Creating skin.nii.gz")
                skin = extract_skin(img_in_orig, nib.load(file_out / "body.nii.gz"))
                nib.save(skin, file_out / "skin.nii.gz")

    return nib.Nifti1Image(img_data, img_pred.affine)
