"""
This package has been taken from TotalSegmentator
https://github.com/wasserth/TotalSegmentator/tree/af44342d7b5a7f6331b33b26434b58882871a68a and modified as follows:
* The bin/TotalSegmentator file was renamed to totalsegmentator/TotalSegmentator.py and the if-else logics for the different tasks has also been stored in totalsegmentator/task_info.py for easier access. Additionally, the command line parameter previous_output was added to be able to read a multi label segmentation that can be used for further subtasks. The logic to read the previous output and compute a mask can be found in lines 397-403.
* The totalsegmentator/store_models_as_onnx.py file was added to store the models as ONNX to be able to use them with Triton.
* The bin/totalseg_combine_masks file was removed.
* The totalsegmentator/measurements.py file was created to store some statistics about the regions.
* The totalsegmentator/util.py file was created to store utility functions for the measurements.
* Converted print statements into logging in:
  * totalsegmentator/alignment.py
  * totalsegmentator/cropping.py
  * totalsegmentator/libs.py
  * totalsegmentator/nnunet.py
  * totalsegmentator/postprocessing.py
  * totalsegmentator/statistics.py
* In totalsegmentator/alignment.py, the functions load_nibabel_image_with_axcodes and convert_nibabel_to_orginal_with_axcodes were added.
* Changed nibabel image to numpy array for the crop_to_mask function in totalsegmentator/cropping.py.
* Returned the weights_path in totalsegmentator/libs.py.
* Added the get_parts_for_regions function to return the correct masks in combine_masks in totalsegmentator/libs.py.
* In totalsegmentator/map_to_binary.py:
  * Uncommented the lung_pleural label for pleural_pericard_effusion in line 326.
  * Added the BCA labels in line 2.
  * Added a dictionary for the labels for which radiomics features should be extracted from line 355.
  * Added a reverse_class_map variable to store the class_map variable in reverse in lines 708-710.
  * Added a reverse_class_map_complete variable to store each reverse class map together with the name of the segmentation task in lines 712-714.
* In totalsegmentator/nnunet.py:
  * The all_in_gpu variable in line 111 was set to true.
  * The parameter resample_only_thickness was added to the nnUNet_predict_image function.
  * The axcodes have been added as an input parameter to the nnUNet_predict_image function.
  * The cropping logic has been slightly changed to also accept numpy arrays as input in lines 227-234.
  * Line 241 was changed to convert an image to any even axcodes, not only to canonical.
  * In lines 242-257 it is possible to resample the image to a specific thickness without having to resample every axes.
  * The force_split check in line 279 was changed to ensure that the image has at least 200px in the z-axis.
  * The resample to original resolution and to the original axes has been changed accordingly to the previous modifications.
  * The postprocessing of the lung_vessels task has been changed in lines 529-542 such that if the original crop input was a numpy array, it can be be used for the postprocessing.
* Added the postprocess_lung_vessels function to totalsegmentator/postprocessing.py to combine the outputs in one single output mask.
* Changed the get_radiomics_features function in totalsegmentator/statistics.py to compute radiomics features for each label separately with appropriate exceptions and removed the list of standard features.


TotalSegmentator is licenced under Apache License 2.0.
A copy of the license text can be found at
https://github.com/wasserth/TotalSegmentator/blob/af44342d7b5a7f6331b33b26434b58882871a68a/LICENSE
"""
