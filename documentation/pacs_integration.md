# PACS Integration

## How does it work?
- The user sends a study to the BOA.
- The study is received by the Orthanc instance, which then creates a task.
- The task is picked up by a task management system, which then starts the segmentations and the computation of the Excel file with the measurements.
- If specified, the segmentations are saved locally and uploaded to the DicomWeb instance.
- If desired, we can provide an additional system where the segmentations can be viewed. Just write us an email.
- If specified, the Excel file is saved locally and uploaded to an SMB share.
- The user can then download the Excel file from the SMB share and look at the segmentations on the DicomWeb instance. The segmentations and the Excel file are also available in the specified local folder.
- If no local output folder was specified, the folder where the computations were originally performed is deleted.

Load the docker images:
```bash
docker pull # Published images coming soon!
```
or clone the repository and build the images. If you are doing this, please set up the environment variables as described above first.
```bash
source scripts/generate_version.sh
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

### Send a study to the BOA
You can then add the instance to your PACS of choice by adding `{YOUR_IP}` and the port `4242` to the location manager to your PACS. Below there is a screenshot of how this looks in Horos.

<div align="center">
  Example in Horos:
  <br>
  <a href="https://horosproject.org/">
    <img src="images/horos.png" alt="Screenshot Horos">
  </a>
</div>

In this case, the IP is the same as the one of my machine because I am testing locally. The AETTitle that you specify will be the name of the folder where the results will be stored, so you that can create different endpoints to computed different cohorts.

## Notes on RabbitMQ
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
