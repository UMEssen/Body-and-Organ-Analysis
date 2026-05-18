import unittest
from pathlib import Path

from body_organ_analysis.cli import run


class TestBOA(unittest.TestCase):
    test_folder = Path.cwd() / "test_images"

    def get_image_file(self, name: str, dicom: bool) -> Path:
        image_dir = self.test_folder / name
        return image_dir / "dicom" if dicom else image_dir / "image.nii.gz"

    def test_dicom_inference_gpu(self) -> None:
        dicom_file = self.get_image_file("cnr", True)
        output_folder = self.test_folder / "output"
        output_folder.mkdir(exist_ok=True)

        run(
            [
                "-i",
                str(dicom_file),
                "-o",
                str(output_folder),
                "-m",
                "body_parts",
                "--fast-bca",
                "--bca-no-pdf",
                "--skip-contrast-information",
            ]
        )

    def test_dicom_inference_cpu(self) -> None:
        pass

    def test_nifti_inference_gpu(self) -> None:
        pass

    def test_nifti_inference_cpu(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
