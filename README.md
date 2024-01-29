<br />
<div align="center">
  <a href="https://ship-ai.ikim.nrw/">
    <img src="images/boa-logo.svg" alt="Logo">
  </a>
</div>

# BOA: Body and Organ Analysis

## What is it?
BOA is a tool for segmentation of CT scans developed by the [SHIP-AI group at the Institute for Artificial Intelligence in Medicine](https://ship-ai.ikim.nrw/). Combining the [TotalSegmentator](https://arxiv.org/abs/2208.05868) and the [Body Composition Analysis](https://pubmed.ncbi.nlm.nih.gov/32945971/), this tool is capable of analyzing medical images and identifying the different structures within the human body, including bones, muscles, organs, and blood vessels.

<div align="center">
    <img src="https://raw.githubusercontent.com/UMEssen/Body-and-Organ-Analysis/main/images/boa.png" alt="BOA">
</div>

The BOA tool can be used to generate full body segmentations of CT scans:

<div align="center">
    <img src="https://raw.githubusercontent.com/UMEssen/Body-and-Organ-Analysis/main/images/segmentation.png" alt="Segmentation of a human body">
</div>

Additionally, the generated segmentations can be used as input to generate realistic images using [Siemens' Cinematic Rendering](https://www.siemens-healthineers.com/digital-health-solutions/cinematic-rendering).

<div align="center">
    <img src="https://raw.githubusercontent.com/UMEssen/Body-and-Organ-Analysis/main/images/cinematic.svg" alt="Cinematic rendering">
</div>

## Citation

If you use this tool, please cite the following papers:

[BOA](https://journals.lww.com/investigativeradiology/abstract/9900/boa__a_ct_based_body_and_organ_analysis_for.176.aspx):
```
Haubold, J., Baldini, G., Parmar, V., Schaarschmidt, B. M., Koitka, S., Kroll, L., van Landeghem, N., Umutlu, L., Forsting, M., Nensa, F., & Hosch, R. (2023). BOA: A CT-Based Body and Organ Analysis for Radiologists at the Point of Care. Investigative radiology, 10.1097/RLI.0000000000001040. Advance online publication. https://doi.org/10.1097/RLI.0000000000001040
```

[TotalSegmentator](https://pubs.rsna.org/doi/10.1148/ryai.230024):
```
Wasserthal J, Breit H-C, Meyer MT, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiol. Artif. Intell. 2023:e230024. Available at: https://pubs.rsna.org/doi/10.1148/ryai.230024.
```
[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z):

```
Isensee F, Jaeger PF, Kohl SAA, et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat. Methods. 2021;18(2):203–211. Available at: https://www.nature.com/articles/s41592-020-01008-z.
```

## How to run?

* Set up the [environment variables](./documentation/environment_variables.md).
* Either use the [PACS integration](./documentation/pacs_integration.md) or the [command line tool](./documentation/command_line.md).

## Notes on Performance

To make an estimate on how much power and time is needed to process a study, we used the [following table](https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/runtime_table.png) provided by the TotalSegmentator. However, for very large series (e.g. 1600 slices 1mm), the performance may be worse and more CPU power may be needed. According to our tests, 16GB of GPU should be sufficient.
