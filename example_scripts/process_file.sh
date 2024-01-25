#!/bin/bash

INPUT_FILE=$1
# TODO: Naming scheme for the study ID, comment/uncomment to change it
# The study ID is the same as the name of the folder it is in
STUDY_ID=$(basename $(dirname $INPUT_FILE))
TT=$(basename $(dirname $(dirname $INPUT_FILE)))
# other option: The study ID is the same as the name of the file
# STUDY_ID=$(basename $INPUT_FILE .nii.gz)
OUTPUT_DIR=$2/$TT/$STUDY_ID
VERSION=$(basename $2)

# TODO: Change this to anything you like
CHECK_NAME="output.xlsx"

if [ -f $OUTPUT_DIR/$CHECK_NAME ]; then
    echo $STUDY_ID already exists
    exit
else
    echo $TT/$STUDY_ID is being computed
fi

WORKING_DIR=$(mktemp -d)
docker run \
    --rm \
    -v $INPUT_FILE:/image.nii.gz \
    -v $WORKING_DIR:/workspace \
    -e NVIDIA_VISIBLE_DEVICES=$GPU_ID \
    --runtime=nvidia \
    --network host \
    --user $(id -u):$(id -g) \
    --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint /bin/sh \
    ship-ai/boa-cli \
    -c \
    "python body_organ_analysis --input-image /image.nii.gz --output-dir /workspace/ --models total+bca --verbose"

if [ -f $WORKING_DIR/$CHECK_NAME ]; then
    echo $STUDY_ID successfully computed
    mkdir -p $OUTPUT_DIR
    cp $WORKING_DIR/* $OUTPUT_DIR/
    cp $INPUT_FILE $OUTPUT_DIR/image.nii.gz
fi
rm -rf $WORKING_DIR
