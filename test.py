from body_organ_analysis.cli import run
from pathlib import Path


# TODO as test method
def is_gzipped(path):
    with open(path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def main() -> None:
    run([
        "--input-image",
        "/local/work/janstraus/boa/test/image.nii.gz",
        "--output-dir",
        "/local/work/janstraus/boa/test/output",
        "--models",
        "all",
        "--verbose",
        "--device",
        "gpu:0",
        "--bca-no-pdf",
        "--skip-contrast-information",
    ])

# TODO remove script
if __name__ == "__main__":
    main()
