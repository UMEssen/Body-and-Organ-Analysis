<br />
<div align="center">
  <a href="https://ship-ai.ikim.nrw/">
    <img src="images/boa-logo.svg" alt="Logo">
  </a>
</div>

# BOA: Body and Organ Analyzer

- [BOA: Body and Organ Analyzer](#boa--body-and-organ-analyzer)
  * [What is it?](#what-is-it-)
  * [How does it work?](#how-does-it-work-)
  * [How to run?](#how-to-run-)
    + [Environment Variables](#environment-variables)
    + [Notes on RabbitMQ](#notes-on-rabbitmq)
    + [Run](#run)
    + [Send a study to the analyzer](#send-a-study-to-the-analyzer)
    + [Notes on Performance](#notes-on-performance)
  * [Outputs](#outputs)
  * [Command Line Tool](#command-line-tool)

## What is it?
BOA is a tool for segmentation of CT scans developed by the [SHIP-AI group at the Institute for Artificial Intelligence in Medicine](https://ship-ai.ikim.nrw/). Combining the [TotalSegmentator](https://arxiv.org/abs/2208.05868) and the [Body Composition Analysis](https://pubmed.ncbi.nlm.nih.gov/32945971/), this tool is capable of analyzing medical images and identifying the different structures within the human body, including bones, muscles, organs, and blood vessels.

If you use this tool, please make sure to cite the following papers: [BCA](https://pubmed.ncbi.nlm.nih.gov/32945971/), [TotalSegmentator](https://arxiv.org/abs/2208.05868) and [nnU-Net](https://www.nature.com/articles/s41592-020-01008-z).

## How does it work?
- The user sends a study to the analyzer.
- The study is received by the Orthanc instance, which then creates a task.
- The task is picked up by a task management system, which then starts the segmentations and the computation of the Excel file with the measurements.
- If specified, the segmentations are saved locally and uploaded to the DicomWeb instance.
- If desired, we can provide an additional system where the segmentations can be viewed. Just write us an email.
- If specified, the Excel file is saved locally and uploaded to an SMB share.
- The user can then download the Excel file from the SMB share and look at the segmentations on the DicomWeb instance. The segmentations and the Excel file are also available in the specified local folder.
- If no local output folder was specified, the folder where the computations were originally performed is deleted.

## How to run?

### Environment Variables
Set up the environment variables by changing the corresponding line in `.env`, if you do not need a specific environment variable, please delete it from `.env`. You can find an example environment file in [.env_sample](./.env_sample).
- `LOCAL_WEIGHTS_PATH`: The local paths where the TotalSegmentator (and BCA) weights should be stored after downloading. It is also possible to remove this variable, and the weights will be stored in the container. However, this means that the weights will be downloaded every time the container is newly created. **Note**: Please create the local directory before if you are not using the root user, this avoids having to change the permissions later. If you want the weights to be stored within the container, you can just remove lines 47 and 80 from [docker-compose.yml](./docker-compose.yml). **Note 2**: The BCA weights are currently not available but are coming soon!
- `RABBITMQ_USERNAME`: Select a username for your user for the task broker, which is going to manage the tasks coming from the Orthanc instance. You can skip this if you already have a RabbitMQ instance running. If you use your instance, please read the [Notes-on-RabbitMQ](#Notes-on-RabbitMQ) section below.
- `RABBITMQ_PASSWORD`: Select a safe password for the broker. You can skip this if you have skipped the step above.
- `CELERY_BROKER`: The default is `amqp://TODO:TODO@rabbitmq/`, where you have to substitute the first TODO with `RABBITMQ_USERNAME`, and the second TODO with `RABBITMQ_PASSWORD`. If you already have a RabbitMQ instance running, you can use that URL instead.
- `ORTHANC_URL`: The URL where our Orthanc is accessible, if you are using the docker-compose it should just be `http://orthanc`.
- `ORTHANC_PORT`: If the port 8042 is already busy, you can specify another port.
- `ORTHANC_USER`: The Orthanc instance should be protected by a username and password. This is not actually needed to use the software, but only to access the Orthanc on the web interface.
- `ORTHANC_PWD`: The password for the Orthanc instance, it is important to set a good password such that the web interface of the Orthanc instance is secure.
- `SEGMENTATION_DIR`: The local directory where the segmentations should be stored. In case you do not specify a folder, or the directory does not exist, a temporary folder is used, which will then be deleted at the end. This means that if both `SEGMENTATION_DIR` and `SEGMENTATION_UPLOAD_URL` are not specified, the segmentations will be lost. **Note**: Please create the directory before if you are not using the root user, this avoids having to change the permissions later.
- `SEGMENTATION_UPLOAD_URL`: Full link to a DicomWeb instance to upload the segmentations, e.g. `http://orthanc:8043/dicom-web/`. If not specified, the segmentations are only saved locally. In this case, it is important to not use our already running Orthanc instance as the target for storing the segmentations. However, another Orthanc instance can be used.
- `UPLOAD_USER`: Username for the DicomWeb instance.
- `UPLOAD_PWD`: Password for the DicomWeb instance.
- `SMB_DIR_OUTPUT`: Link to an SMB share where the final output Excel should be saved.
- `SMB_USER`: Username for the SMB share.
- `SMB_PWD`: Password for the SMB share.
- `NVIDIA_ID`: The ID of the GPU that you want to use. It can also be multiple values separated by a comma e.g. `1,2`.
- `DOCKER_USER`: The ID of the user such that the container can be run in user mode and that the outputs can be easily accessed. You can get it with `$(id -u):$(id -g)`. It can also be run in root mode (by specifying `root`), but then the outputs will be owned by root and it may be needed to change the ownership.
- `PATIENT_INFO_IN_OUTPUT`: This option changes how the output folder structure looks like. If this variable is `True`, then the folder structure will look like the following:
   - AET (the name that you have given to this endpoint in your PACS, more on this later)
      - PatientName_PatientSurname_DateOfBirth
         - StudyDate_AccessionNumber_StudyDescription
            - SeriesNumber_SeriesDescription
               - image.nii.gz
               - input_dicoms
                  - SOP1.dcm
                  - ...
               - AccessionNumber_SeriesNumber_SeriesDescription.xlsx
               - total.nii.gz
               - body-regions.nii.gz
               - ...

   otherwise, the folder structure will look like this
   - AET (the name that you have given to the endpoint in your PACS, more on this later)
      - StudyDate_AccessionNumber_StudyDescription
        - SeriesNumber_SeriesDescription
            - image.nii.gz
            - input_dicoms
               - SOP1.dcm
               - ...
            - AccessionNumber_SeriesNumber_SeriesDescription.xlsx
            - total.nii.gz
            - body-regions.nii.gz
            - ...

   The `PATIENT_INFO_IN_OUTPUT="true"` option has the drawback that:
   - The folder structure is not unique if your DICOMs are anonymized, i.e. if you have two patients with the same anonymized name and birthdate (e.g. John Doe, 1970), then you will have clashes.
   - The output folders will not be anonymized and will contain the patient name and birthdate.

### Notes on RabbitMQ
RabbitMQ always checks whether the tasks have been received by the consumer, and if the consumer takes too long, the tasks are going to be killed. This is a problem in our case, because the tasks may take more time to complete, and if 40-50 studies have been sent at the same time, they are going to be killed by RabbitMQ because they have not yet been completed. This can be set with the `timout_consumer` variable of RabbitMQ, and since this highly depends on the amount of studies that are being sent in one go, we decided to disable this variable. This is done by creating a file in `/etc/rabbitmq/advanced.config` with the following content.

```
%% advanced.config
[
  {rabbit, [
    {consumer_timeout, undefined}
  ]}
].
```

If you are using your own RabbitMQ instance, please also use this setting, or contact us if you have a better idea!

### Run
Load the docker images:
```bash
docker pull # Published images coming soon!
```
or clone the repository and build the images
```bash
docker compose build orthanc rabbitmq worker-gpu
```

Download the [docker-compose.yml](./docker-compose.yml) file and run the following command
```bash
docker compose up orthanc rabbitmq worker-gpu -d
```
with `worker-gpu` if you want to use a local GPU and `worker` if you have Triton instance running.  Remove `rabbitmq` in case you already have an instance running.

You can also just
```bash
docker compose up -d
```

!!!**IMPORTANT**!!!: if you are using windows, substitute the `docker-compose` with `docker -f docker-compose-win.yml`. `docker-compose-win.yml` has not been tested extensively so if you have any problems please contact us!

### Send a study to the analyzer
You can then add the instance to your PACS of choice by adding `{YOUR_IP}` and the port `4242` to the location manager to your PACS. Below there is a screenshot of how this looks in Horos.

<div align="center">
  Example in Horos:
  <br>
  <a href="https://horosproject.org/">
    <img src="images/horos.png" alt="Screenshot Horos">
  </a>
</div>

In this case, the IP is the same as the one of my machine because I am testing locally. The AETTitle that you specify will be the name of the folder where the results will be stored, so you that can create different endpoints to computed different cohorts.

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

## Command Line Tool

Additionally, the BOA can also be run from the command line to compute all the segmentations in one go and without connecting to the PACS.

First, get the image.
```bash
docker pull # Published images coming soon!
```

or clone the repository and build the image
```bash
docker build -t ship-ai/boa-cli --file scripts/cli.dockerfile .
```

then you can run your image!

### Run on Linux
```bash
docker run \
    --rm \
    -v $INPUT_FILE:/image.nii.gz \
    -v $WORKING_DIR:/workspace \
    -v $LOCAL_WEIGHTS_PATH:/app/weights \ # Add this to avoid redownloading
    --runtime=nvidia \
    --network host \
    --user $(id -u):$(id -g) \ # This sets your user ID
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/sh \
    ship-ai/boa-cli \
    -c \
    "python body_organ_analyzer --input-image /image.nii.gz --output-dir /workspace/ --models all --verbose"
```

### Run on Windows
```bash
docker run \
    --rm \
    -v $INPUT_FILE:/image.nii.gz \
    -v $WORKING_DIR:/workspace \
    -v $LOCAL_WEIGHTS_PATH:/app/weights \ # Add this to avoid redownloading
    --gpus all \
    --network host \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/sh \
    ship-ai/boa-cli \
    -c \
    "python body_organ_analyzer --input-image /image.nii.gz --output-dir /workspace/ --models all --verbose"
```

where `$INPUT_FILE` is the path to the input CT and `$WORKING_DIR` is the path to the directory where the results will be stored.

The command can be customized with the following options:
* In `--models` you can specify which models you want to run: `total` for the TotalSegmentator, `bca` for the Body Composition Analysis, `total+bca` for both or `all` for all the possible models: `body`, `total`, `lung_vessels`, `cerebral_bleed`, `hip_implant`, `coronary_arteries`, `pleural_pericard_effusion`, `liver_vessels`, `bca`.
* You can also specify whether you want to extract the radiomics features by adding `--radiomics`. **CAREFUL**: This has currently not been tested extensively, so it might not work as expected.
* During the process some segmentations are generated, which are then postprocessed and the original segmentations are deleted. If you want all versions of the segmentations, you can add `--keep-debug-segmentations`.
* There are other parameters that either belong to the BCA or to the TotalSegmentator, which you can view in [`cli.py`](cli.py).
