#!/bin/bash

# Source and destination directories
SOURCE_DIR="."
DEST_DIR="merra-2"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist: $SOURCE_DIR"
  exit 1
fi

# Check if destination directory exists, if not create it
if [ ! -d "$DEST_DIR" ]; then
  mkdir -p "$DEST_DIR"
fi

# Loop through files in the source directory
for file in "$SOURCE_DIR"/*; do
  # Check if it's a file (not a directory)
  if [ -f "$file" ]; then
    # Extract the base filename
    filename=$(basename "$file")

    # Check if the filename ends with '=true'
    if [[ "$filename" == *=true ]]; then
      # Remove everything from the question mark to the end
      new_filename=$(echo "$filename" | sed 's/\?.*//')

      # Move and rename the file
      mv "$file" "$DEST_DIR/$new_filename"
      echo "Moved and renamed: $filename -> $new_filename"
    fi
  fi
done

echo "All files have been processed."