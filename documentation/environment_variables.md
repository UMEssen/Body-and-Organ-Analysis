# Environment Variables
Set up the environment variables by changing the corresponding line in `.env`, if you do not need a specific environment variable, please delete it from `.env`. You can find an example environment file in [.env_sample](./.env_sample).
- `LOCAL_WEIGHTS_PATH`: The local paths where the TotalSegmentator (and BCA) weights should be stored after downloading. It is also possible to remove this variable, and the weights will be stored in the container. However, this means that the weights will be downloaded every time the container is newly created. **Note**: Please create the local directory before if you are not using the root user, this avoids having to change the permissions later. If you want the weights to be stored within the container, you can just remove lines 47 and 80 from [docker-compose.yml](./docker-compose.yml).
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
