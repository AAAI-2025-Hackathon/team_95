import os

# Get the directory of the current script (pre-processing folder)
script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)


data_folder_path = os.path.join(root_directory, 'data')
print(f"Path to the data folder: {data_folder_path}")

def analyze_fire_folders(root_dir):
    """
    Analyzes fire folders organized by year to count fire occurrences and observations.

    Args:
        root_dir (str): The root directory containing year folders (e.g., "2018", "2019", etc.).
    """

    yearly_fire_counts = {}
    fire_event_observations = {}

    for year_folder in os.listdir(root_dir):
        if year_folder.isdigit() and year_folder in ["2018", "2019", "2020", "2021"]:
            year_path = os.path.join(root_dir, year_folder)
            if os.path.isdir(year_path):
                fire_events_in_year = 0
                fire_event_observations[year_folder] = {}
                for fire_folder in os.listdir(year_path):
                    fire_folder_path = os.path.join(year_path, fire_folder)
                    if os.path.isdir(fire_folder_path):
                        fire_events_in_year += 1
                        tiff_files_count = 0
                        for filename in os.listdir(fire_folder_path):
                            if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):
                                tiff_files_count += 1
                        fire_event_observations[year_folder][fire_folder] = tiff_files_count
                yearly_fire_counts[year_folder] = fire_events_in_year

    print("Total fire occurrences per year:")
    for year, count in yearly_fire_counts.items():
        print(f"Year {year}: {count} fire events")

    print("\nNumber of observations (TIFF files) per fire event:")
    for year, events in fire_event_observations.items():
        print(f"Year {year}:")
        for event, observation_count in events.items():
            print(f"  - Fire Event: {event}: {observation_count} observations")

if __name__ == '__main__':
    analyze_fire_folders(root_directory)
