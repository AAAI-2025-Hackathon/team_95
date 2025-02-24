import rasterio
import matplotlib.pyplot as plt

def view_tiff_with_rasterio(tiff_filepath, pixel_row, pixel_col):
    """
    Reads and displays a geospatial TIFF image using rasterio and matplotlib,
    prints band names (if available), prints a sample of band values,
    and extracts and prints values for a specific pixel across all bands.

    Args:
        tiff_filepath (str): The path to the geospatial TIFF file.
        pixel_row (int): The row index of the pixel (starting from 0).
        pixel_col (int): The column index of the pixel (starting from 0).
    """
    try:
        # Open the TIFF file with rasterio
        with rasterio.open(tiff_filepath) as src:
            # Read the raster data
            raster_data = src.read() # Reads all bands into a 3D NumPy array (bands, rows, cols)

            # Band names 
            band_names = [
                "VIIRS band M11", "VIIRS band I2", "VIIRS band I1", "NDVI", "EVI2",
                "total precipitation", "wind speed", "wind direction",
                "minimum temperature", "maximum temperature", "energy release component",
                "specific humidity", "slope", "aspect", "elevation",
                "Palmer drought severity index", "landcover class",
                "forecast total precipitation", "forecast wind speed", "forecast wind direction",
                "forecast temperature", "forecast specific humidity", "active fire"
            ]

            # Basic raster information
            print("CRS:", src.crs)
            print("Bounds:", src.bounds)
            print("Resolution:", src.res)
            print("Shape:", src.shape) # Shape will now reflect (bands, height, width)
            print("Number of Bands:", src.count)
            print("Data Type:", src.dtypes)


            # **Get and print pixel values for the specified pixel across all bands**
            print(f"\nPixel Values at Row: {pixel_row}, Column: {pixel_col}:")
            if 0 <= pixel_row < raster_data.shape[1] and 0 <= pixel_col < raster_data.shape[2]: # Check if pixel coordinates are valid
                pixel_values = raster_data[:, pixel_row, pixel_col] # Extract values for all bands at the pixel
                for i in range(src.count): # Iterate through bands and band names
                    if i < len(band_names):
                        band_name = band_names[i]
                    else:
                        band_name = f"Band {i+1}" 
                    print(f"  Band {i+1} ({band_name}): Value = {pixel_values[i]}")
            else:
                print("  Error: Pixel coordinates are out of image bounds.")
            print("\n--- End of Pixel Values ---")


            # Display the first band as an image (for visualization)
            plt.imshow(raster_data[0], cmap='gray') # Display band 1
            plt.title("Raster Image (Band 1)")
            plt.colorbar()
            plt.show()

    except FileNotFoundError:
        print(f"Error: TIFF file not found at '{tiff_filepath}'")
    except rasterio.errors.RasterioIOError as rio_error:
        print(f"Rasterio Error: Could not open or read TIFF file. {rio_error}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    tiff_file_path = '/s/lovelace/h/nobackup/sangmi/hackathon/2018/fire_21458798/2018-01-01.tif'
    view_tiff_with_rasterio(tiff_file_path, 0, 0)
