# Command Line Tool

Additionally, the BOA can also be run from the command line to compute all the segmentations in one go and without connecting to the PACS.

First, get the image.
```bash
docker pull # Published images coming soon!
```

or clone the repository and build the image
```bash
source scripts/generate_version.sh
docker build -t ship-ai/boa-cli --file scripts/cli.dockerfile .
```

then you can run your image!

## Run on Linux
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
    "python body_organ_analysis --input-image /image.nii.gz --output-dir /workspace/ --models all --verbose"
```

## Run on Windows
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
    "python body_organ_analysis --input-image /image.nii.gz --output-dir /workspace/ --models all --verbose"
```

where `$INPUT_FILE` is the path to the input CT and `$WORKING_DIR` is the path to the directory where the results will be stored.

The command can be customized with the following options:
* In `--models` you can specify which models you want to run: `total` for the TotalSegmentator, `bca` for the Body Composition Analysis, `total+bca` for both or `all` for all the possible models: `body`, `total`, `lung_vessels`, `cerebral_bleed`, `hip_implant`, `coronary_arteries`, `pleural_pericard_effusion`, `liver_vessels`, `bca`.
* You can also specify whether you want to extract the radiomics features by adding `--radiomics`. **CAREFUL**: This has currently not been tested extensively, so it might not work as expected.
* During the process some segmentations are generated, which are then postprocessed and the original segmentations are deleted. If you want all versions of the segmentations, you can add `--keep-debug-segmentations`.
* There are other parameters that either belong to the BCA or to the TotalSegmentator, which you can view in [`cli.py`](cli.py).
