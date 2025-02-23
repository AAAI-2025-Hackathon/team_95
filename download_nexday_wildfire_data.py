import kagglehub

# Download latest version
path = kagglehub.dataset_download("fantineh/next-day-wildfire-spread")

print("Path to dataset files:", path)