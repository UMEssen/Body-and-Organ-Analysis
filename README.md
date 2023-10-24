<br />
<div align="center">
  <a href="https://ship-ai.ikim.nrw/">
    <img src="images/boa-logo.svg" alt="Logo">
  </a>
</div>

# BOA: Body and Organ Analysis

- [BOA: Body and Organ Analysis](#boa--body-and-organ-analysis)
  * [What is it?](#what-is-it-)
  * [How does it work?](#how-does-it-work-)
  * [How to run?](#how-to-run-)
    + [Environment Variables](#environment-variables)
    + [Notes on RabbitMQ](#notes-on-rabbitmq)
    + [Run](#run)
    + [Send a study to the BOA](#send-a-study-to-the-boa)
    + [Notes on Performance](#notes-on-performance)
  * [Outputs](#outputs)
  * [Command Line Tool](#command-line-tool)

## What is it?
BOA is a tool for segmentation of CT scans developed by the [SHIP-AI group at the Institute for Artificial Intelligence in Medicine](https://ship-ai.ikim.nrw/). Combining the [TotalSegmentator](https://arxiv.org/abs/2208.05868) and the [Body Composition Analysis](https://pubmed.ncbi.nlm.nih.gov/32945971/), this tool is capable of analyzing medical images and identifying the different structures within the human body, including bones, muscles, organs, and blood vessels.


If you use this tool, please cite the following papers:

```
Haubold J, Baldini G, Parmar V, et al. A CT-Based Body and Organ Analysis for Radiologists at the Point of Care. Invest. Radiol. (In Press).
```

```
Wasserthal J, Breit H-C, Meyer MT, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiol. Artif. Intell. 2023:e230024. Available at: https://pubs.rsna.org/doi/10.1148/ryai.230024.
```

```
Isensee F, Jaeger PF, Kohl SAA, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. Methods. 2021;18(2):203–211. Available at: https://www.nature.com/articles/s41592-020-01008-z.
```

## How to run?

* Set up the [environment variables](./documentation/environment_variables.md).
* Either use the [PACS integration](./documentation/pacs_integration.md) or the [command line tool](./documentation/command_line_tool.md).

### Notes on Performance

To make an estimate on how much power and time is needed to process a study, we used the [following table](https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/runtime_table.png) provided by the TotalSegmentator. However, for very large series (e.g. 1600 slices 1mm), the performance may be worse and more CPU power may be needed. According to our tests, 16GB of GPU should be absolutely sufficient.

## Outputs

The outputs of BOA are listed below, all of them will appear in the `SEGMENTATION_DIR` folder, some of them will be uploaded to SMB and some will be uploaded with DicomWeb, if they are configured.
Currently, the produced DICOM-segs have placeholders for the anatomical names of the tissues.
- Segmentations (BCA and TotalSegmentator), present in `SEGMENTATION_DIR` and uploaded to DicomWeb (optional).
  - Total Body Segmentation (`total.nii.gz`): Segmentation of 104 body regions ([TotalSegmentator](https://arxiv.org/abs/2208.05868)).
  - Intracerebral Hemorrhage Segmentation (`cerebral_bleed.nii.gz`).
  - Lung Vessels and Airways Segmentation (`lung_vessels_airways.nii.gz`): Segmentation of trachea/bronchia/airways ([paper](https://www.sciencedirect.com/science/article/pii/S0720048X22001097)).
  - Liver Vessels and Tumor Segmentation (`liver_vessels.nii.gz`).
  - Hip Implant Segmentation (`hip_implant.nii.gz`).
  - Coronary Arteries Segmentation (`coronary_arteries.nii.gz`): Segmentation of the coronary arteries.
  - Pleural Pericardial Effusion Segmentation (`pleural_pericard_effusion.nii.gz`): pleural effusion ([paper](https://journals.lww.com/investigativeradiology/Fulltext/2022/08000/Automated_Detection,_Segmentation,_and.8.aspx)), pericardial effusion (cite [paper](https://www.mdpi.com/2075-4418/12/5/1045)).
  - Body Regions Segmentation (`body-regions.nii.gz`): Segmentation of the body regions ([BCA](https://pubmed.ncbi.nlm.nih.gov/32945971/)).
  - Body Parts Segmentation (`body-parts.nii.gz`): Segmentation of body and extremities.
  - Tissue Segmentation (`tissues.nii.gz`): Segmentation of the tissues ([BCA](https://pubmed.ncbi.nlm.nih.gov/32945971/)).
- Measurements/Reports, present in `SEGMENTATION_DIR` and uploaded to SMB (optional).
  - `AccessionNumber_SeriesNumber_SeriesDescription.xlsx`: Excel file with all the measurements from the BCA and the TotalSegmentator. The `info` sheet contains general information about the patient, such as IDs, dates, DICOM tags, contrast phase, the BOA version. The `region_statistics` sheet has information about the segmentations that were computed together with their volume and some other statistical information. `bca-aggregated-measurements` contains all the measurements for the aggregated regions of the BCA, which are also visible in the report. `bca-slice-measurements` contains information about the volume of each tissue for each slice of the CT scan. `bca-slice-measurements-no-limbs` contains the same information, but the extremities are removed from the computation.
  - `report.pdf`: Report of the BCA findings.
  - `preview_total.png`: Preview of the TotalSegmentator segmentation.
- Other files containing measurements are also stored in the output directory (in `.json` format), and are not uploaded anywhere else. These measurements all appear in the resulting Excel report.
