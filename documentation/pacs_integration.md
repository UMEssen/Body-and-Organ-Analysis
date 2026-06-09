# PACS Integration

This guide covers the automated, PACS-driven deployment of BOA. For one-off
segmentation of a single study, use the command line tool documented in the
[README](../README.md#command-line-tool).

## How it works

- A user sends a study to BOA from their PACS.
- The study is received by the **Orthanc** instance, which fires a Python
  on-change callback when the series becomes stable. The callback filters the
  series (CT, axial, ≥ 10 instances) and queues a task.
- The task is picked up by **RabbitMQ** and processed by a **Celery worker**
  (GPU or CPU), which runs the segmentations and builds the Excel measurements.
- The results are saved to the local output folder, and — if configured —
  uploaded to a DicomWeb instance (segmentations as DICOM-SEG) and an SMB share
  (Excel + report PDF).
- The original series is then removed from Orthanc, and per-task statistics are
  written to the optional monitoring database.
- If no local output folder is available and neither SMB nor DicomWeb is
  configured, the task aborts because the results would not be retrievable.

## 1. Prepare the images

Set the `GIT_VERSION` and `PACKAGE_VERSION` build variables (baked into the
images for debugging):

```bash
source scripts/generate_version.sh
```

Pull the prebuilt images:

```bash
docker pull shipai/boa-orthanc
docker pull shipai/boa-rabbitmq
docker pull shipai/boa-worker-gpu

# Only needed for CPU-only inference (no GPU)
docker pull shipai/boa-worker-cpu
```

or build them from the repository (configure the environment first, see below):

```bash
docker compose build orthanc rabbitmq worker-gpu
```

> **Windows.** GPU access in [docker-compose.yml](../docker-compose.yml) uses
> the cross-platform Compose device-reservation syntax, so the **same** compose
> file works on Linux and on Windows (Docker Desktop / WSL2) — there is no
> longer a separate `docker-compose-win.yml`. Building the images on Windows
> can be unreliable; prefer the prebuilt images from Docker Hub.

## 2. Configure the environment

Copy the sample and edit it:

```bash
cp .env_sample .env
```

All runtime configuration lives in `.env`. The full reference is in
[Environment variables](#environment-variables) below. At minimum, set the
RabbitMQ and Orthanc credentials (and `CELERY_BROKER` to match), choose the
GPU index (`NVIDIA_ID`), and decide where outputs go (`SEGMENTATION_DIR` and/or
the SMB / DicomWeb variables).

## 3. Start the stack

```bash
docker compose up -d orthanc rabbitmq worker-gpu
```

Use `worker-cpu` instead of `worker-gpu` if you do not have a local GPU; it runs
the same segmentations on the CPU (slower) — enabling the fast modes
(`FAST_TOTAL=true` / `FAST_BCA=true`) is strongly recommended there. Drop
`rabbitmq` if you already operate your own
broker (see [Notes on RabbitMQ](#notes-on-rabbitmq)).

To additionally enable the monitoring database, start it as well:

```bash
docker compose up -d orthanc rabbitmq worker-gpu monitoring
```

The GPU is pinned via `NVIDIA_ID` (the host GPU index) through the
`device_ids` reservation in the compose file. The monitoring database is
exposed on host port `5433` by default; change it in the compose file if that
port is taken. This port does not affect functionality — the database is
reached over the internal Docker network.

## 4. Send a study to BOA

Add BOA to your PACS as a DICOM node using your host's IP and port **4242**.
The DICOM **AET** you configure becomes the top-level folder name for the
results, so you can create different endpoints for different cohorts.

### Example in Horos

![Screenshot Horos](../images/horos.png)

Here the IP is the local machine because the test runs locally.

## Environment variables

Booleans accept `1`/`true` (case-insensitive); empty values and the literal
`TODO` placeholder are treated as unset.

### Stack-wide

| Variable | Default | Purpose |
| --- | --- | --- |
| `DOCKER_USER` | `root` | `uid:gid` the worker drops to (via the entrypoint's chown + gosu) so outputs are owned by you. Use `$(id -u):$(id -g)`. |
| `NVIDIA_ID` | `0` | Host GPU index, used by the `device_ids` GPU reservation in the compose file. |
| `LOCAL_WEIGHTS_PATH` | `./weights` | Host path mounted to `/app/weights` (`TOTALSEG_WEIGHTS_PATH`). Caches the model weights between runs. Create it beforehand if you are not running as root. |
| `SEGMENTATION_DIR` | `./output` | Host path mounted to `/storage_directory` for local outputs. If it does not exist, a temporary folder is used and the results are lost unless an SMB / DicomWeb sink is configured. |

### Worker

| Variable | Default | Purpose |
| --- | --- | --- |
| `PACS_MODEL` | `all` | `+`-separated model list (e.g. `total+bca`) or `all`. See [Models](../README.md#models). Invalid names are dropped with a warning. |
| `FAST_BCA` | `False` | Use the single-fold BCA variant instead of the 5-fold ensemble. |
| `FAST_TOTAL` | `False` | Run TotalSegmentator in fast mode. |
| `PATIENT_INFO_IN_OUTPUT` | `False` | Insert a `PatientName_PatientBirthDate` layer into the output folder path. **Privacy-sensitive** (see below). |
| `CELERY_BROKER` | `amqp://TODO:TODO@rabbitmq/` | AMQP URL. Substitute the two `TODO`s with `RABBITMQ_USERNAME` / `RABBITMQ_PASSWORD`. |
| `TRITON_URL` | (unset) | gRPC URL of a Triton inference server. **Triton is not available yet**; `worker-cpu` runs plain CPU inference regardless of this value. |

> The deprecated `PREDICT_FAST` variable no longer has any effect on the worker
> — use `FAST_BCA` / `FAST_TOTAL` instead.

`PATIENT_INFO_IN_OUTPUT="true"` changes the output layout to include the patient
name and birth date:

```text
AET/PatientName_PatientBirthDate/StudyDate_AccessionNumber_StudyDescription/SeriesNumber_SeriesDescription/
```

instead of the anonymized default:

```text
AET/StudyDate_AccessionNumber_StudyDescription/SeriesNumber_SeriesDescription/
```

The drawback is that output folders then contain patient identifiers, and the
layout is not unique if your DICOMs are anonymized (two "John Doe, 1970"
patients would collide).

### Orthanc

| Variable | Default | Purpose |
| --- | --- | --- |
| `ORTHANC_URL` | `http://orthanc` | Base URL the worker uses to reach Orthanc inside the Docker network. |
| `ORTHANC_PORT` | `8042` | Host port mapped to Orthanc's HTTP UI. |
| `ORTHANC_USERNAME` | `TODO` | HTTP basic-auth user; becomes `ORTHANC__REGISTERED_USERS`. |
| `ORTHANC_PASSWORD` | `TODO` | HTTP basic-auth password. Set a strong value — it protects the web UI. |

### RabbitMQ

| Variable | Default | Purpose |
| --- | --- | --- |
| `RABBITMQ_USERNAME` | `TODO` | Broker user (`RABBITMQ_DEFAULT_USER`). Used in `CELERY_BROKER`. |
| `RABBITMQ_PASSWORD` | `TODO` | Broker password (`RABBITMQ_DEFAULT_PASS`). |

### Monitoring (optional)

Failures are non-fatal: if any of these is missing or the database is
unreachable, monitoring is silently skipped and segmentation still runs.

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_HOST` | `monitoring` | Service DNS name of the database. |
| `POSTGRES_PORT` | `5432` | Internal port (host-mapped to `5433`). |
| `POSTGRES_USER` | `boa_user` | Database user (created by `init.sql`). |
| `POSTGRES_PASSWORD` | `TODO` | Database password. |
| `POSTGRES_DATABASE` | `ship_ai_boa` | Database name (matches `init.sql`). |
| `POSTGRES_DATA` | `./postgres` | Host path mounted to the database's data directory. **Always set this**, even without monitoring, or Compose will complain. |

### Output sinks (optional)

Configure either, both, or neither. Each group must be fully set (and not
`TODO`) for that upload to fire.

| Variable | Purpose |
| --- | --- |
| `SMB_DIR_OUTPUT` | UNC-style path of an SMB share for the Excel + report PDFs. |
| `SMB_USER` / `SMB_PWD` | SMB credentials. |
| `SEGMENTATION_UPLOAD_URL` | DicomWeb STOW endpoint for the segmentation DICOM-SEGs, e.g. `http://orthanc:8043/dicom-web/`. Do not point this at the receiving Orthanc instance. |
| `UPLOAD_USER` / `UPLOAD_PWD` | DicomWeb credentials. |

## Monitoring

When the monitoring database is enabled, every task upserts a row into the
`boa_entries` table (schema in [init.sql](../init.sql)), keyed by the Celery
`task_id`. This data can drive a
[Grafana dashboard](https://grafana.com/grafana/dashboards/). You can also point
the `POSTGRES_*` variables at an existing database — just apply the statements
from `init.sql` there first.

## Notes on RabbitMQ

RabbitMQ kills a task if the consumer takes longer than its
`consumer_timeout`. BOA tasks can run for a long time (and many studies may be
queued at once), so we raise this limit substantially. We also permit the
deprecated features that Celery's pidbox queues still rely on under RabbitMQ
4.x. This is configured in `/etc/rabbitmq/advanced.config`, baked into the
`shipai/boa-rabbitmq` image:

```erlang
%% advanced.config
[
  {rabbit, [
    {consumer_timeout, 86400000},
    {permit_deprecated_features, #{transient_nonexcl_queues => true, global_qos => true}}
  ]}
].
```

If you run your own RabbitMQ instance, apply the same settings.

## Outputs

All outputs appear in `SEGMENTATION_DIR`; some are additionally uploaded via
SMB and DicomWeb when configured. The produced DICOM-SEGs currently use
placeholders for the anatomical names of the tissues.

- **Segmentations** (`SEGMENTATION_DIR` + optional DicomWeb):
  - `total.nii.gz` — Total Body Segmentation
    ([TotalSegmentator](https://arxiv.org/abs/2208.05868)).
  - `cerebral_bleed.nii.gz` — intracerebral hemorrhage.
  - `lung_vessels_airways.nii.gz` — lung vessels and airways
    ([paper](https://www.sciencedirect.com/science/article/pii/S0720048X22001097)).
  - `liver_vessels.nii.gz` — liver vessels and tumor.
  - `hip_implant.nii.gz` — hip implant.
  - `pleural_pericard_effusion.nii.gz` — pleural
    ([paper](https://journals.lww.com/investigativeradiology/Fulltext/2022/08000/Automated_Detection,_Segmentation,_and.8.aspx))
    and pericardial
    ([paper](https://www.mdpi.com/2075-4418/12/5/1045)) effusion.
  - `heartchambers_highres.nii.gz` — high-resolution heart chambers.
  - `body_regions.nii.gz` — body regions
    ([BCA](https://pubmed.ncbi.nlm.nih.gov/32945971/)).
  - `body_parts.nii.gz` — body and extremities.
  - `tissues.nii.gz` — tissue segmentation
    ([BCA](https://pubmed.ncbi.nlm.nih.gov/32945971/)).
- **Measurements / reports** (`SEGMENTATION_DIR` + optional SMB):
  - `AccessionNumber_SeriesNumber_SeriesDescription.xlsx` — all measurements
    from BCA and TotalSegmentator. See
    [Outputs](../README.md#outputs) for the sheet-by-sheet breakdown.
  - `report.pdf` — BCA report.
  - `preview_total.png` — preview of the TotalSegmentator output.
- Additional `*-measurements.json` files are written to the output directory;
  their contents also appear in the Excel workbook.
