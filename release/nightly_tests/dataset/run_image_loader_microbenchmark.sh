#!/bin/bash


# Exit if any of the test commands fail.
set -e pipeline

INPUT_DIR=~/imagenet-1gb
OUTPUT_DIR=~/imagenet-1gb-data

# Download 1GB dataset from S3 to local disk.
aws s3 sync s3://imagenetmini-1000-1gb $INPUT_DIR

# Preprocess files to get to the directory structure that torch dataloader
# expects.
mkdir -p $OUTPUT_DIR
mv $INPUT_DIR/train/* $OUTPUT_DIR
# for fullname in $(find $INPUT_DIR -type f); do
#     filename=$(basename "$fullname")
#     class_dir=$(echo "$filename" | awk '{split($0, array, "_"); print array[1]}')
#     img_path=$(echo "$filename" | awk '{split($0, array, "_"); print array[2]}')
#     mkdir -p "$OUTPUT_DIR"/"$class_dir"
#     out_path="$OUTPUT_DIR/$class_dir/$img_path"
#     echo "$out_path"
#     cp "$fullname" "$out_path"
# done

python image_loader_microbenchmark.py
