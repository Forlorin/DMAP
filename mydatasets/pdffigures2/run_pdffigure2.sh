#!/bin/bash
# sbt "runMain org.allenai.pdffigures2.FigureExtractorBatchCli /path/to/your/file -s stat_file.json -m /path/to/your/file/image -d /path/to/your/file/data"

# check arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <input_dir> <stat_file> <image_output_dir> <data_output_dir>"
    exit 1
fi
export SBT_OPTS="-Xms2g -Xmx32g -XX:+UseG1GC"
INPUT_DIR="$1"
STAT_FILE="$2"
IMAGE_DIR="$3"
DATA_DIR="$4"
WORK_DIR="$5"

cd $WORK_DIR

# run sbt command
sbt "runMain org.allenai.pdffigures2.FigureExtractorBatchCli \
  $INPUT_DIR \
  -s $STAT_FILE \
  -m $IMAGE_DIR \
  -d $DATA_DIR" \

