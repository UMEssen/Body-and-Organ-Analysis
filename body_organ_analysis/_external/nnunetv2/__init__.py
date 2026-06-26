"""
Original source:
    https://github.com/MIC-DKFZ/nnUNet/tree/273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0

Original tagged release:
    https://github.com/MIC-DKFZ/nnUNet/tree/v2.6.4

Modifications:
    - nnunetv2/training/nnUNetTrainer/variants/training_length/
      nnUNetTrainer_Xepochs_NoMirroring.py

      Added class nnUNetTrainer_1500epochs_NoMirroring, based on
      nnUNetTrainer_2000epochs_NoMirroring, with num_epochs changed to 1500
      for the Dataset543_BCA_body_parts nnU-Net model.

nnU-Net is licensed under the Apache License, Version 2.0.

A copy of the Apache License 2.0 should be included with this project.
The original nnU-Net license can be found at:
    https://github.com/MIC-DKFZ/nnUNet/blob/273d6147f2d71bafc27a5e1c6fcab9f59ceba7d0/LICENSE
"""
