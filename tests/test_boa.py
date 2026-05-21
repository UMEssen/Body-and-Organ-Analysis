import platform
import shutil
import unittest
from pathlib import Path

from body_organ_analysis.cli import run


class TestBOA(unittest.TestCase):
    test_folder = Path.cwd() / "test_images"

    def get_image_file(self, name: str, dicom: bool) -> Path:
        image_dir = self.test_folder / name
        return image_dir / "dicom" if dicom else image_dir / "image.nii.gz"

    @unittest.skipUnless(
        platform.system() in {"Windows", "Linux"}, "Requires Windows or Linux"
    )
    def test_dicom_inference_gpu(self) -> None:
        dicom_file = self.get_image_file("cnr", True)
        output_folder = self.test_folder / "output"
        shutil.rmtree(output_folder, ignore_errors=True)
        output_folder.mkdir()

        run(
            [
                "-i",
                str(dicom_file),
                "-o",
                str(output_folder),
                "-m",
                "bca",
                "-d",
                "gpu:0",
                "--cnr-adjustment",
                "--bca-no-pdf",
                "--skip-contrast-information",
            ]
        )

    # def test_dicom_inference_cpu(self) -> None:
    #     dicom_file = self.get_image_file("cnr", True)
    #     output_folder = self.test_folder / "output"
    #     shutil.rmtree(output_folder, ignore_errors=True)
    #     output_folder.mkdir()

    #     run(
    #         [
    #             "-i",
    #             str(dicom_file),
    #             "-o",
    #             str(output_folder),
    #             "-m",
    #             "bca",
    #             "-d",
    #             "cpu",
    #             "--fast-bca",
    #             "--fast-total",
    #             "--bca-no-pdf",
    #             "--skip-contrast-information",
    #         ]
    #     )

    # Not working yet
    # RuntimeError: ConvTranspose 3D is not supported on MPS
    # def test_dicom_inference_mps(self) -> None:
    #     dicom_file = self.get_image_file("cnr", True)
    #     output_folder = self.test_folder / "output"
    #     shutil.rmtree(output_folder, ignore_errors=True)
    #     output_folder.mkdir()

    #     run(
    #         [
    #             "-i",
    #             str(dicom_file),
    #             "-o",
    #             str(output_folder),
    #             "-m",
    #             "bca",
    #             "-d",
    #             "mps",
    #             "--fast-bca",
    #             "--fast-total",
    #             "--bca-no-pdf",
    #             "--skip-contrast-information",
    #         ]
    #     )

    def test_nifti_inference_gpu(self) -> None:
        pass

    def test_nifti_inference_cpu(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
