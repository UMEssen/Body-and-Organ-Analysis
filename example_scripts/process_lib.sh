INPUT_DIR=/path/to/the/folder/with/inputs
OUTPUT_DIR=/path/to/the/folder/for/outputs

mkdir -p $OUTPUT_DIR

SCRIPT_DIR=$(dirname $0)

for d in /data/*/ ; do
  $SCRIPT_DIR/process_file.sh $d $OUTPUT_DIR
done
