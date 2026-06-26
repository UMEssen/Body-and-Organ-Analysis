from pathlib import Path

TEST_FOLDER = Path(__file__).parent.parent / "test_images"
CNR_DICOM_DIR = TEST_FOLDER / "cnr" / "dicom"
CNR_NIFTI_FILE = TEST_FOLDER / "cnr" / "image.nii.gz"
OUTPUT_GPU_DIR = TEST_FOLDER / "output_gpu"
OUTPUT_CPU_DIR = TEST_FOLDER / "output_cpu"
