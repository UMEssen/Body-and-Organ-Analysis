from body_organ_analysis.cli import run


# TODO remove script
if __name__ == "__main__":
    run([
        "--input-image",
        "/local/work/janstraus/boa/test/image.nii.gz",
        "--output-dir",
        "/local/work/janstraus/boa/test/output",
        "--models",
        "bca",
        "--verbose",
        "--bca-no-pdf",
        "--skip-contrast-information",
    ])
