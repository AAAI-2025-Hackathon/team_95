#!/bin/bash



# Folder containing the filenames

FOLDER="merra-2"  # Replace with the path to your folder



# Base URL template

BASE_URL="https://huggingface.co/ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M/resolve/main/merra-2/"



# Iterate over all files in the folder

for file in "$FOLDER"/*; do

    # Extract the filename (last part of the path)

    filename=$(basename "$file")

    

    # Construct the full URL by replacing the filename in the base URL

    download_url="${BASE_URL}${filename}?download=true"

    

    # Download the file using wget

    echo "Downloading $filename..."

    wget "$download_url"

    

    # Check if the download was successful

    if [ $? -eq 0 ]; then

        echo "Downloaded $filename successfully."

    else

        echo "Failed to download $filename."

    fi

done



echo "All files processed."
