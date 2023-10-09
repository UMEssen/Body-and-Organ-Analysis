"""
This package has been taken from nnU-Net
https://github.com/MIC-DKFZ/nnUNet/tree/aa53b3b87130ad78f0a28e6169a83215d708d659 and modified as follows:
* The following folders have been removed:
  * nnunet/dataset_conversion
  * nnunet/experiment_planning/old
* nnUNet/nnunet/inference/segmentation_export.py, lines 223-224 were changed such that "force_separate_z" is only printed in the verbose setting
* Custom trainer from the TotalSegmentator were added in nnunet/training/custom_trainers (nnUNetTrainerV2_ep4000.py, nnUNetTrainerV2_ep4000_nomirror.py, nnUNetTrainerV2_ep4000_nomirror.py from https://github.com/wasserth/nnUNet_cust/commit/21deddae237fb9719d86d2883c8ff57a6d109994)
* Some changes have been done to make the models compatible with Triton:
  * nnunet/inference/inference_model.py was added to communicate with the Triton server
  * nnunet/training/model_restore.py was modified to swap the models with the TritonNetwork from nnunet/inference/inference_model.py. In particular, the substitute_model_with_triton function was added in line 126. Additionally, load_model_and_checkpoint_files was changed from line 223 to substitute the normal network with a TritonNetwork.
  * nnunet/inference/predict.py was modified to reset the iteration of Triton models whenever if a cross-validation model is stored as multiple model version (lines 282-283, 507-508, 692-693)
  * nnunet/training/network_training/nnUNetTrainer.py was modified in lines 409-410 to update the version if a cross-validation model is stored as multiple model versions.
* In line 727 of nnunet/training/network_training/nnUNetTrainer.py was changed to enforce the conversion to tuple of self.patch_size
* In line 472-473 the self.print_to_log_file call was converted to a simple print statement to be able to mute the print in the TotalSegmentator

nnU-Net is licenced under Apache License 2.0.
A copy of the license text can be found at
https://github.com/MIC-DKFZ/nnUNet/blob/aa53b3b87130ad78f0a28e6169a83215d708d659/LICENSE
"""

from __future__ import absolute_import

print(
    "\n\nPlease cite the following paper when using nnUNet:\n\nIsensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "
    '"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." '
    "Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z\n\n"
)
print(
    "If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet\n"
)

from . import *
