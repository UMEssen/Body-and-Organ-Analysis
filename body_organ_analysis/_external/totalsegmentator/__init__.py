"""
Original source:
    https://github.com/wasserth/TotalSegmentator/tree/dde221478d81221f0d9df558c29f8f29d8f8f3b7

Original tagged release:
    https://github.com/wasserth/TotalSegmentator/tree/v2.12.0

Modifications:
    - totalsegmentator/map_to_binary.py

      Imported the BodyParts, BodyRegion and Tissue enums from
      body_composition_analysis and added "body_parts", "body_regions" and
      "tissue" entries to class_map, each mapping every enum member's value to
      its lower-cased name. This exposes the Body Composition Analysis label
      maps through TotalSegmentator's class_map.

    - totalsegmentator/dicom_io.py

      Removed the dicom2nifti-based dcm_to_nifti() function and its
      "import dicom2nifti". DICOM-to-NIfTI conversion is performed upstream by
      BOA, so the dicom2nifti dependency is no longer used here.

    - totalsegmentator/nnunet.py

      Dropped the dcm_to_nifti import and removed the in-function DICOM
      conversion branch of nnUNet_predict_image (which called dcm_to_nifti),
      matching the removal in dicom_io.py; the input is expected as NIfTI.

      Imported load_nibabel_image_with_axcodes from body_composition_analysis.io
      and added the resample_only_thickness (default False) and axcodes
      (default "RAS") parameters to nnUNet_predict_image. When
      resample_only_thickness is set, the image is reoriented to the given
      axcodes and only the slice thickness is resampled, preserving the
      original in-plane spacing.

TotalSegmentator is licensed under the Apache License, Version 2.0.

A copy of the Apache License 2.0 should be included with this project.
The original TotalSegmentator license can be found at:
    https://github.com/wasserth/TotalSegmentator/blob/dde221478d81221f0d9df558c29f8f29d8f8f3b7/LICENSE
"""
