"""
This package has been taken from TotalSegmentator
https://github.com/wasserth/TotalSegmentator/tree/af44342d7b5a7f6331b33b26434b58882871a68a and modified as follows:
* The bin/TotalSegmentator file was renamed to totalsegmentator/TotalSegmentator.py and the if-else logics for the different tasks has also been stored in totalsegmentator/task_info.py for easier access. Additionally, the command line parameter previous_output was added to be able to read a multi label segmentation that can be used for further subtasks. The logic to read the previous output and compute a mask can be found in lines 397-403.
* The totalsegmentator/store_models_as_onnx.py file was added to store the models as ONNX to be able to use them with Triton.
* The bin/totalseg_combine_masks file was removed.
* Converted print statements into logging in:
  * totalsegmentator/alignment.py
  * totalsegmentator/cropping.py
  * totalsegmentator/libs.py
  * totalsegmentator/nnunet.py
  * totalsegmentator/postprocessing.py
  * totalsegmentator/statistics.py
  * totalsegmentator/preview.py
* In totalsegmentator/preview.py:
    * Commented the part of the code using the slice actor (lines 256-263) because it causes a memory leak.
    * Removed the ct_img parameter from plot_roi_group
    * Commented unused imports
* In totalsegmentator/alignment.py, the functions load_nibabel_image_with_axcodes and convert_nibabel_to_orginal_with_axcodes were added.
* Changed nibabel image to numpy array for the crop_to_mask function in totalsegmentator/cropping.py.
* In totalsegmentator/libs.py:
  * Returned the weights_path in download_pretrained_weights.
  * Added the get_parts_for_regions function to return the correct masks in combine_masks.
  * Updated the WEIGHTS_URL from Zenodo to GitHub for download_pretrained_weights.
  * Add the BCA weights.
  * Fix the download_url_and_unpack function to be able to download from GitHub and Zenodo with the most current changes: https://github.com/wasserth/TotalSegmentator/blob/96335b6e7d60383a2b4710e74e4b08b8c33f3ab9/totalsegmentator/libs.py#L40
* In totalsegmentator/map_to_binary.py:
  * Uncommented the lung_pleural label for pleural_pericard_effusion in line 326.
  * Added the BCA labels in line 2.
  * Added a dictionary for the labels for which radiomics features should be extracted.
  * Added a reverse_class_map variable to store the class_map variable in reverse.
  * Added a reverse_class_map_complete variable to store each reverse class map together with the name of the segmentation task.
  * This file was also updated with the code from this version (https://github.com/wasserth/TotalSegmentator/blob/87560f329b3bd97543f975ebd47ba269933f43ee/totalsegmentator/nnunet.py)
* In totalsegmentator/nnunet.py:
  * The all_in_gpu variable in line 113 was set to true.
  * CT clipping was added to lines 240-242 to ensure that the values of the CT are in the correct range.
  * In lines 289-292 some properties of the niftis were changed to avoid the "ITK ERROR: ITK only supports orthonormal direction cosines.  No orthonormal definition found!" error.
  * The parameter resample_only_thickness was added to the nnUNet_predict_image function.
  * The axcodes have been added as an input parameter to the nnUNet_predict_image function.
  * The cropping logic has been slightly changed to also accept numpy arrays as input in lines 246-257.
  * Line 261 was changed to convert an image to any even axcodes, not only to canonical.
  * In lines 262-277 it is possible to resample the image to a specific thickness without having to resample every axes.
  * The force_split check in line 301 was changed to ensure that the image has at least 200px in the z-axis.
  * The resample to original resolution and to the original axes has been changed accordingly to the previous modifications.
  * The postprocessing of the lung_vessels task has been changed in lines 572-584 such that if the original crop input was a numpy array, it can be be used for the postprocessing.
  * In line 204 and 591 the parameter "compute_skin" was added to make it optional.
  * This file was also updated with the code from this version (https://github.com/wasserth/TotalSegmentator/blob/87560f329b3bd97543f975ebd47ba269933f43ee/totalsegmentator/nnunet.py)
* Added the postprocess_lung_vessels function to totalsegmentator/postprocessing.py to combine the outputs in one single output mask.
* Changed the get_radiomics_features function in totalsegmentator/statistics.py to compute radiomics features for each label separately with appropriate exceptions and removed the list of standard features.


TotalSegmentator is licenced under Apache License 2.0.
A copy of the license text can be found at
https://github.com/wasserth/TotalSegmentator/blob/af44342d7b5a7f6331b33b26434b58882871a68a/LICENSE
"""
